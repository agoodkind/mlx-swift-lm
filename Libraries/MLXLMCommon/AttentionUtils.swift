import Foundation
import MLX

// ── TurboKV Telemetry ────────────────────────────────────────────────────────
// Feeds C atomics in moe_stream_op.cpp; stats appear in the 10s SSD Stream log:
//   [⚡️ SSD Stream] 7330 MB/s | ... | 🗜 TurboKV 4244t 4.3x
// No per-event prints — zero log spam.

// Direct link to the C atomic recorder in moe_stream_op.cpp
@_silgen_name("mlx_turbo_kv_record")
func _mlxTurboKVRecord(_ tokens: UInt64, _ origBytes: UInt64, _ packedBytes: UInt64)

enum TurboKVTelemetry {
    /// Feed the 10s log aggregator from the cache compression path.
    ///
    /// `keys` and `values` are the packed uint8 arrays just produced by turboQuantEncode.
    /// Shape: [B, nKVH, newTokens, packedDim]
    /// origBytes = B × nKVH × newTokens × headDim × 2B (fp16) × 2 (K+V)
    static func logOnce(compressedOffset: Int, keys: MLXArray, values: MLXArray, headDim: Int) {
        let B = keys.dim(0)
        let nKVH = keys.dim(1)
        let newTokens = keys.dim(2)  // only the tokens just encoded, not cumulative
        let packedBytes = UInt64(keys.nbytes + values.nbytes)
        let origBytes = UInt64(B * nKVH * newTokens * headDim * 2 * 2)  // K+V fp16
        _mlxTurboKVRecord(UInt64(newTokens), origBytes, packedBytes)
    }

    /// Feed the 10s log aggregator from the AttentionUtils decode path.
    static func record(tokens: Int, origBytes: Int, packedBytes: Int) {
        _mlxTurboKVRecord(UInt64(tokens), UInt64(origBytes), UInt64(packedBytes))
    }
}

/// Alias used by KVCache.swift on the compression path.
typealias TurboKVCacheTelemetry = TurboKVTelemetry
// ─────────────────────────────────────────────────────────────────────────────

/// Cache-native attention that can reuse an owner's KV without appending it again.
package protocol SharedAttentionKVCache: KVCache {
    func updateForAttention(keys: MLXArray, values: MLXArray)
    func attend(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray
}

/// An owner's attention presentation, valid until its next cache update.
/// Keep this within one forward pass; serialization state is not an attention view.
package enum AttentionKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int, bits: Int, mode: QuantizationMode
    )
    case native(cache: any SharedAttentionKVCache, sequenceLength: Int)

    package var sequenceLength: Int {
        switch self {
        case .regular(let keys, _): keys.dim(2)
        case .quantized(let keys, _, _, _, _): keys.0.dim(2)
        case .native(_, let sequenceLength): sequenceLength
        }
    }

    package static func update(keys: MLXArray, values: MLXArray, cache: KVCache?) -> Self {
        if let cache = cache as? SharedAttentionKVCache {
            cache.updateForAttention(keys: keys, values: values)
            return .native(cache: cache, sequenceLength: cache.offset)
        } else if let cache = cache as? QuantizedKVCacheProtocol {
            let (keys, values) = cache.updateQuantized(keys: keys, values: values)
            return .quantized(
                keys: keys, values: values,
                groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)
        } else if let cache {
            let (keys, values) = cache.update(keys: keys, values: values)
            let (fullKeys, fullValues) = restoredAttentionHistory(
                cache: cache, keys: keys, values: values)
            return .regular(keys: fullKeys, values: fullValues)
        } else {
            return .regular(keys: keys, values: values)
        }
    }

    package func attend(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        switch self {
        case .regular(let keys, let values):
            return regularAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        case .quantized(let keys, let values, let groupSize, let bits, let mode):
            return quantizedScaledDotProductAttention(
                queries: queries, quantizedKeys: keys, quantizedValues: values,
                scale: scale, mask: mask, groupSize: groupSize, bits: bits, mode: mode)
        case .native(let cache, let sequenceLength):
            // Native storage must still describe the owner's presentation.
            precondition(cache.offset == sequenceLength)
            return cache.attend(queries: queries, scale: scale, mask: mask)
        }
    }
}

/// Whether attention can be split around a plain `KVCache.update` call.
///
/// Quantized and TurboQuant caches own their complete attention operation, so
/// compiled model segments must leave those cache routes on the general path.
package func usesPlainAttentionCacheRoute(_ cache: KVCache) -> Bool {
    if let cache = cache as? KVCacheSimple, cache.turboQuantEnabled { return false }
    return !(cache is QuantizedKVCacheProtocol) && !(cache is SharedAttentionKVCache)
        && !(cache is KVCacheAttentionProtocol)
}

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Detects cache-native attention implementations
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
/// Fallback for scaledDotProductAttention when running on a CPU device
private func fallbackScaledDotProductAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    // Handle Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
    var k = keys
    var v = values
    let qHeads = queries.dim(1)
    let kHeads = keys.dim(1)
    if qHeads > kHeads {
        let repeats = qHeads / kHeads
        k = MLX.repeated(k, count: repeats, axis: 1)
        v = MLX.repeated(v, count: repeats, axis: 1)
    }

    var scores = (queries * scale).matmul(k.transposed(0, 1, 3, 2))
    if let maskArray = mask.mask {
        scores = scores + maskArray
    }
    let softMaxScores = MLX.softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
    return matmul(softMaxScores, v)
}

private func restoredAttentionHistory(
    cache: KVCache, keys: MLXArray, values: MLXArray
) -> (MLXArray, MLXArray) {
    var fullKeys = keys
    var fullValues = values
    if let kvCache = cache as? KVCacheSimple,
        let pk = kvCache.polarKeys,
        let pv = kvCache.polarValues,
        kvCache.compressedOffset > 0
    {
        let historyK = MLXFast.turboDecodeK(packed: pk).asType(keys.dtype)
        let historyV = MLXFast.turboDecodeV(packed: pv).asType(values.dtype)
        var mergedK = historyK
        var mergedV = historyV
        if kvCache.turboSplitHeads {
            let B = historyK.dim(0)
            let H2 = historyK.dim(1)
            let T = historyK.dim(2)
            mergedK = historyK.reshaped(B, H2 / 2, T, 512)
            mergedV = historyV.reshaped(B, H2 / 2, T, 512)
        }
        fullKeys = concatenated([mergedK, keys], axis: 2)
        fullValues = concatenated([mergedV, values], axis: 2)
    }

    return (fullKeys, fullValues)
}

private func regularAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let mask = attentionMask(mask, sequenceLength: keys.dim(2))
    if Device.defaultDevice().deviceType == .cpu {
        return fallbackScaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }
    return MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: scale, mask: mask)
}

private func attentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode, sequenceLength: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    guard case .array(let customMask) = mask, customMask.dim(-1) > sequenceLength else {
        return mask
    }
    switch customMask.ndim {
    case 2: return .array(customMask[0..., ..<sequenceLength])
    case 4: return .array(customMask[0..., 0..., 0..., ..<sequenceLength])
    default: preconditionFailure("Unsupported mask dimensionality: \(customMask.ndim)")
    }
}

public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
        return regularAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let turboCache = cache as? TurboQuantKVCache {
        let L = queries.dim(2)
        if L > 1 && !turboCache.isCompressed {
            // Keep prefill storage uncompressed until the first decode.
            let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
            return regularAttention(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale, mask: mask
            )
        }
        // Compressed storage requires the cache-owned attention operation.
        return turboCache.compressedAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
    } else if let attentionCache = cache as? KVCacheAttentionProtocol {
        return attentionCache.updateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask)
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        precondition(
            Device.defaultDevice().deviceType != .cpu,
            "Quantized KV attention requires a GPU device.")
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: attentionMask(mask, sequenceLength: quantizedKeys.0.dim(2)),
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        let (fullKeys, fullValues) = restoredAttentionHistory(
            cache: cache, keys: cachedKeys, values: cachedValues)
        return regularAttention(
            queries: queries,
            keys: fullKeys,
            values: fullValues,
            scale: scale,
            mask: mask
        )
    }
}

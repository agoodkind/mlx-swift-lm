import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4

private enum Gemma4Error: LocalizedError {
    case multimodalTokenCountMismatch(kind: String, featureTokens: Int, promptTokens: Int)
    case imagePlaceholderMismatch(images: Int, placeholders: Int)
    case missingVideoTokenId
    case missingVideoTokenMapping
    case videoTokenIdMismatch(configuration: Int, tokenizer: Int)
    case missingVideoBoundaryTokenMapping(configurationKey: String, tokenId: Int)
    case emptyVideoFrames
    case videoPlaceholderCountMismatch(videos: Int, placeholders: Int)

    var errorDescription: String? {
        switch self {
        case .missingVideoTokenId:
            return "Gemma4 video input requires video_token_id in the model configuration."
        case .missingVideoTokenMapping:
            return "Gemma4 video input requires a tokenizer video token."
        case .videoTokenIdMismatch(let configuration, let tokenizer):
            return "Gemma4 video_token_id is \(configuration), but the tokenizer uses \(tokenizer)."
        case .missingVideoBoundaryTokenMapping(let configurationKey, let tokenId):
            return
                "Gemma4 video input requires a tokenizer mapping for \(configurationKey) \(tokenId)."
        case .emptyVideoFrames:
            return "Gemma4 video input must contain at least one frame."
        case .videoPlaceholderCountMismatch(let videos, let placeholders):
            return
                "Gemma4 video input contains \(videos) videos and \(placeholders) prompt placeholders."
        case .multimodalTokenCountMismatch(let kind, let featureTokens, let promptTokens):
            return
                "Gemma4 \(kind) token count mismatch: encoder produced \(featureTokens) soft tokens, but the prompt contains \(promptTokens) \(kind) tokens."
        case .imagePlaceholderMismatch(let images, let placeholders):
            return
                "Gemma4 image placeholder mismatch: the request has \(images) images but the prompt contains at least \(placeholders) image placeholders."
        }
    }
}

private func gemma4BuildLayerTypes(hiddenLayers: Int, slidingWindowPattern: Int) -> [String] {
    let pattern =
        Array(repeating: "sliding_attention", count: max(slidingWindowPattern - 1, 0))
        + ["full_attention"]
    guard !pattern.isEmpty else { return Array(repeating: "full_attention", count: hiddenLayers) }
    var result: [String] = []
    result.reserveCapacity(hiddenLayers)
    while result.count < hiddenLayers {
        result.append(contentsOf: pattern)
    }
    return Array(result.prefix(hiddenLayers))
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
func gemma4DefaultTextRopeParameters() -> [String: [String: StringOrNumber]] {
    [
        "full_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(1_000_000.0),
            "rope_type": .string("proportional"),
        ],
        "sliding_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(10_000.0),
            "rope_type": .string("default"),
        ],
    ]
}

private func gemma4MaskedScatter(
    inputTensor: MLXArray, mask: MLXArray, source: MLXArray
) -> MLXArray {
    let flattenedInput = inputTensor.flattened()
    let flattenedMask = mask.flattened().asArray(Bool.self)
    let flattenedSource = source.flattened()

    let targetIndices = flattenedMask.enumerated().compactMap { idx, value in
        value ? Int32(idx) : nil
    }
    guard !targetIndices.isEmpty else {
        return inputTensor
    }

    guard flattenedSource.shape[0] == targetIndices.count else {
        fatalError(
            "Masked scatter shape mismatch. source=\(flattenedSource.shape[0]) mask=\(targetIndices.count)"
        )
    }

    let result = flattenedInput
    result[MLXArray(targetIndices, [targetIndices.count])] = flattenedSource
    return result.reshaped(inputTensor.shape)
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
enum Gemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            return keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            return keys.0.dim(-2)
        }
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
func gemma4AdjustAttentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    keyLength: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mask {
    case .array(let maskArray):
        guard let maskLength = maskArray.shape.last, maskLength > keyLength else {
            return mask
        }
        let start = maskLength - keyLength
        return .array(maskArray[.ellipsis, start...])
    case .arrays, .causal, .none:
        return mask
    }
}

private func gemma4TokenTypeIds(
    inputIds: MLXArray,
    imageTokenId: Int?,
    videoTokenId: Int?,
    audioTokenId: Int?
) -> MLXArray {
    var tokenTypeIds = MLXArray.zeros(like: inputIds).asType(.int32)
    if let imageTokenId {
        tokenTypeIds = MLX.where(inputIds .== imageTokenId, MLXArray(1), tokenTypeIds)
    }
    if let videoTokenId {
        tokenTypeIds = MLX.where(inputIds .== videoTokenId, MLXArray(2), tokenTypeIds)
    }
    if let audioTokenId {
        tokenTypeIds = MLX.where(inputIds .== audioTokenId, MLXArray(3), tokenTypeIds)
    }
    return tokenTypeIds
}

private func gemma4TextOnlyPromptTokens(_ input: LMInput) -> MLXArray {
    let tokens = input.text.tokens
    if tokens.ndim == 2, tokens.dim(0) == 1 {
        return tokens[0]
    }
    if tokens.ndim == 1 {
        return tokens
    }
    return tokens.flattened()
}

private func gemma4PrepareTextOnly(
    _ input: LMInput,
    cache: [any KVCache],
    prefill: PrefillParameters,
    languageModel: Gemma4TextLanguageModel
) throws -> PrepareResult {
    let y = gemma4TextOnlyPromptTokens(input).expandedDimensions(axis: 0)
    let convertedCache = cache.map { $0 }
    let totalPositions = y.dim(1)

    let processed = try prefill.forEachChunk(total: totalPositions) { range in
        _ = languageModel(y[0..., range], cache: convertedCache)
        asyncEval(cache)
    }
    if processed > 0 { eval(cache) }
    let result = languageModel(y[0..., processed...], cache: convertedCache)
    prefill.progress?(totalPositions, totalPositions)
    return .logits(result)
}

private func gemma4BlockSequenceIdsForMask(_ tokenTypeIds: MLXArray) -> MLXArray {
    let isVision = (tokenTypeIds .== 1) | (tokenTypeIds .== 2)
    let sequenceLength = isVision.dim(1)
    guard sequenceLength > 0 else {
        return MLXArray.zeros(like: tokenTypeIds).asType(.int32) - 1
    }

    let previous =
        if sequenceLength == 1 {
            MLXArray.zeros(like: isVision)
        } else {
            concatenated(
                [
                    MLXArray.zeros(like: isVision[0..., ..<1]),
                    isVision[0..., ..<(sequenceLength - 1)],
                ],
                axis: 1
            )
        }
    let starts = isVision & logicalNot(previous)
    let groupIds = cumsum(starts.asType(.int32), axis: 1) - 1
    return MLX.where(isVision, groupIds, MLXArray.zeros(like: groupIds) - 1)
}

private func gemma4ApplyBlockwiseBidirectionalOverlay(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    tokenTypeIds: MLXArray,
    sequenceLength: Int,
    windowSize: Int?
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let baseMask: MLXArray
    switch mask {
    case .array(let array):
        baseMask = array
    case .arrays(let arrays):
        guard let first = arrays.first else {
            return mask
        }
        baseMask = first
    case .causal:
        baseMask = createCausalMask(n: sequenceLength, offset: 0, windowSize: windowSize)
    case .none:
        baseMask = MLXArray.ones([sequenceLength, sequenceLength], dtype: .bool)
    }

    guard tokenTypeIds.dim(1) == baseMask.dim(-1) else {
        return mask
    }

    let blockSequenceIds = gemma4BlockSequenceIdsForMask(tokenTypeIds)
    let queryBlocks = expandedDimensions(blockSequenceIds, axis: -1)
    let keyBlocks = expandedDimensions(blockSequenceIds, axis: -2)
    let sameBlock = (queryBlocks .!= -1) & (queryBlocks .== keyBlocks)
    return .array(baseMask | expandedDimensions(sameBlock, axis: 1))
}

private func gemma4CompactPrefixRows(features: MLXArray, validMask: MLXArray) -> MLXArray {
    let maskRows = validMask.asArray(Bool.self)
    let batch = validMask.dim(0)
    let length = validMask.dim(1)
    var rows: [MLXArray] = []
    rows.reserveCapacity(batch)

    for batchIdx in 0 ..< batch {
        let start = batchIdx * length
        let count = maskRows[start ..< start + length].reduce(0) { $0 + ($1 ? 1 : 0) }
        if count > 0 {
            rows.append(features[batchIdx, ..<count, 0...])
        }
    }

    guard !rows.isEmpty else {
        return features.reshaped(-1, features.dim(-1))[..<0, 0...]
    }
    return concatenated(rows, axis: 0)
}
// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let globalKVHeads: Int?
    public let headDim: Int
    public let globalHeadDim: Int
    public let vocabularySize: Int
    public let vocabularySizePerLayerInput: Int
    public let numKVSharedLayers: Int
    public let hiddenSizePerLayerInput: Int
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTraditional: Bool
    public let finalLogitSoftcapping: Float?
    public let useDoubleWideMLP: Bool
    public let enableMoEBlock: Bool
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    public let attentionKEqV: Bool
    public let useBidirectionalAttention: String?
    public let layerTypes: [String]
    public let ropeParameters: [String: [String: StringOrNumber]]
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case vocabularySize = "vocab_size"
        case vocabularySizePerLayerInput = "vocab_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTraditional = "rope_traditional"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionKEqV = "attention_k_eq_v"
        case useBidirectionalAttention = "use_bidirectional_attention"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_text"
        let isUnified = modelType == "gemma4_unified_text" || modelType == "gemma4_unified"
        hiddenSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize)
            ?? (isUnified ? 3840 : 1536)
        hiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers)
            ?? (isUnified ? 48 : 35)
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize)
            ?? (isUnified ? 15_360 : 6144)
        attentionHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads)
            ?? (isUnified ? 16 : 8)
        kvHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.kvHeads) ?? (isUnified ? 8 : 1)
        globalKVHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalKVHeads)
            ?? (isUnified ? 1 : nil)
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalHeadDim) ?? 512
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySize) ?? 262_144
        vocabularySizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySizePerLayerInput)
            ?? vocabularySize
        numKVSharedLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.numKVSharedLayers)
            ?? (isUnified ? 0 : 20)
        hiddenSizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSizePerLayerInput)
            ?? (isUnified ? 0 : 256)
        slidingWindow =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindow)
            ?? (isUnified ? 1024 : 512)
        slidingWindowPattern =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindowPattern)
            ?? (isUnified ? 6 : 5)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxPositionEmbeddings) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        ropeTraditional =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.ropeTraditional) ?? false
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: CodingKeys.finalLogitSoftcapping) ?? 30.0
        useDoubleWideMLP =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useDoubleWideMLP) ?? true
        enableMoEBlock =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.enableMoEBlock) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.numExperts)
        topKExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.topKExperts)
        moeIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: CodingKeys.moeIntermediateSize)
        attentionKEqV =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.attentionKEqV) ?? isUnified
        useBidirectionalAttention =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.useBidirectionalAttention)
            ?? (isUnified ? "vision" : nil)
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: CodingKeys.ropeParameters)
            ?? (isUnified
                ? [
                    "full_attention": [
                        "partial_rotary_factor": .float(0.25),
                        "rope_theta": .float(1_000_000.0),
                        "rope_type": .string("proportional"),
                    ],
                    "sliding_attention": [
                        "rope_theta": .float(10_000.0),
                        "rope_type": .string("default"),
                    ],
                ] : gemma4DefaultTextRopeParameters())
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: CodingKeys.layerTypes)
            ?? gemma4BuildLayerTypes(
                hiddenLayers: hiddenLayers, slidingWindowPattern: slidingWindowPattern)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings) ?? true
    }
}

/// Configuration of the Gemma 4 audio encoder (`model_type: gemma4_audio`).

public struct Gemma4Configuration: Codable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4VisionConfiguration
    public let modelType: String
    public let quantization: BaseConfiguration.Quantization?
    public let imageTokenId: Int
    public let audioTokenId: Int?
    public let videoTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let visionSoftTokensPerImage: Int
    public let visionSoftTokensPerVideoFrame: Int
    public let videoFrameChunkSize: Int
    public let tieWordEmbeddings: Bool
    public let audioConfig: Gemma4AudioConfiguration?

    private let _vocabularySize: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int { _vocabularySize ?? textConfiguration.vocabularySize }
    public var hiddenSize: Int { _hiddenSize ?? textConfiguration.hiddenSize }
    public var padTokenId: Int { _padTokenId ?? 0 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case quantization
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case visionSoftTokensPerVideoFrame = "vision_soft_tokens_per_video_frame"
        case videoFrameChunkSize = "video_frame_chunk_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case audioConfig = "audio_config"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(
            Gemma4TextConfiguration.self, forKey: CodingKeys.textConfiguration)
        visionConfiguration = try c.decode(
            Gemma4VisionConfiguration.self, forKey: CodingKeys.visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4"
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: CodingKeys.quantization)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId)
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId)
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId)
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerImage)
            ?? visionConfiguration.defaultOutputLength
        visionSoftTokensPerVideoFrame =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerVideoFrame) ?? 70
        videoFrameChunkSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoFrameChunkSize) ?? 4
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings)
            ?? textConfiguration.tieWordEmbeddings
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._vocabularySize)
        _hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._hiddenSize)
        _padTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys._padTokenId)
        audioConfig = try c.decodeIfPresent(
            Gemma4AudioConfiguration.self, forKey: CodingKeys.audioConfig)
    }
}

// MARK: - Text

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
typealias Gemma4RMSNormNoScale = MLXLMCommon.Gemma4RMSNormNoScale

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
typealias Gemma4RMSNormZeroShift = MLXLMCommon.Gemma4RMSNormZeroShift

final class Gemma4TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        let isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0
        let useDoubleWide = config.useDoubleWideMLP && isKVSharedLayer
        let hiddenDimensions = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma4TextRouter: Module {
    let topKExperts: Int
    let config: Gemma4TextConfiguration
    private let rootSize: Float

    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts, let topKExperts = config.topKExperts else {
            fatalError("Gemma4 MoE router requires numExperts and topKExperts")
        }

        self.topKExperts = topKExperts
        self.config = config
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let normed = MLXFast.rmsNorm(
            x, weight: (scale * rootSize).asType(x.dtype), eps: config.rmsNormEps)

        let scores = proj(normed)

        let topKIndices = MLX.argPartition(scores, kth: -topKExperts, axis: -1)[
            .ellipsis, (-topKExperts)...,
        ]
        var topKWeights = MLX.takeAlong(scores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1)
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

final class Gemma4TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts,
            let moeIntermediateSize = config.moeIntermediateSize
        else {
            fatalError("Gemma4 MoE experts require numExperts and moeIntermediateSize")
        }

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = x.dim(2)
        let topK = topKIndices.dim(-1)

        let expertOutput = switchGLU(
            x.reshaped(batch * length, hidden),
            topKIndices.reshaped(batch * length, topK)
        )
        let weights = topKWeights.reshaped(batch * length, topK).asType(expertOutput.dtype)
        return weightedExpertSum(expertOutput, weights).reshaped(batch, length, hidden)
    }
}

final class Gemma4ScaledLinear: Module, UnaryLayer {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.scalar = scalar
        self._weight.wrappedValue = MLXArray.zeros([outFeatures, inFeatures])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x.matmul(weight.transposed())) * scalar
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
/// Use `kvSharedOnly: true` in the constructor to skip building local K/V
/// projections (the drafter consumes the target's K/V via `sharedKV` instead).
final class Gemma4TextAttention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float
    let isKVSharedLayer: Bool
    let useKEqV: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale?
    @ModuleInfo var rope: OffsetLayer

    init(config: Gemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.headDim =
            layerType == "full_attention" && config.globalHeadDim > 0
            ? config.globalHeadDim : config.headDim
        self.numHeads = config.attentionHeads
        self.useKEqV = config.attentionKEqV && !isSliding
        self.numKVHeads =
            useKEqV ? (config.globalKVHeads ?? config.kvHeads) : config.kvHeads
        self.scale = 1.0

        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self.isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        // KV-shared layers (the last `num_kv_shared_layers`) reuse an earlier layer's
        // K/V and own no k_proj/v_proj/k_norm/v_norm — the checkpoint ships none, so the
        // module tree must not declare them (else loadWeights fails keyNotFound on the
        // shared layers, e.g. E2B layer 15 / E4B layer 24). `kvSharedOnly` is the drafter's
        // always-shared variant; `isKVSharedLayer` covers the target's own shared tail.
        // Mirrors the text backbone (MLXLLM/Models/Gemma4Text.swift).
        if !kvSharedOnly && !isKVSharedLayer {
            self._kProj.wrappedValue = Linear(
                config.hiddenSize, numKVHeads * headDim, bias: false)
            if !useKEqV {
                self._vProj.wrappedValue = Linear(
                    config.hiddenSize, numKVHeads * headDim, bias: false)
            }
            self._kNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: headDim, eps: config.rmsNormEps)
            self._vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)
        }
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: headDim, eps: config.rmsNormEps)

        let ropeKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeConfig = config.ropeParameters[ropeKey]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000 : 1_000_000)
        self._rope.wrappedValue = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        queries = qNorm(queries)

        let currentOffset: Int
        let kvState: Gemma4SharedKVState?

        if let sharedKV {
            currentOffset = offset ?? 0
            kvState = sharedKV
        } else {
            // KV-owning path: K/V projections must be present. If they are nil
            // here the layer is KV-shared (drafter `kvSharedOnly`, or the target's
            // shared tail `isKVSharedLayer`) and the caller forgot to pass
            // `sharedKV` — a configuration bug.
            guard let kProj, let kNorm, let vNorm else {
                fatalError(
                    "Gemma4 attention called without sharedKV on a KV-shared layer")
            }
            currentOffset = cache?.offset ?? 0
            var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
            var values =
                if useKEqV {
                    keys
                } else {
                    vProj!(x).reshaped(batch, length, numKVHeads, headDim)
                }
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            values = vNorm(values).transposed(0, 2, 1, 3)
            keys = rope(keys, offset: currentOffset)
            if let quantizedCache = cache as? QuantizedKVCacheProtocol {
                let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(
                    keys: keys, values: values)
                kvState = .quantized(
                    keys: quantizedKeys,
                    values: quantizedValues,
                    groupSize: quantizedCache.groupSize,
                    bits: quantizedCache.bits,
                    mode: quantizedCache.mode
                )
            } else {
                if let cache {
                    (keys, values) = cache.update(keys: keys, values: values)
                }
                kvState = .regular(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: currentOffset)

        guard let kvState else {
            fatalError("Gemma4 attention expected a KV state")
        }
        let localMask = gemma4AdjustAttentionMask(mask, keyLength: kvState.sequenceLength)

        let output: MLXArray =
            switch kvState {
            case .regular(let keys, let values):
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: localMask
                )
            case .quantized(let keys, let values, let groupSize, let bits, let mode):
                quantizedScaledDotProductAttention(
                    queries: queries,
                    quantizedKeys: keys,
                    quantizedValues: values,
                    scale: scale,
                    mask: localMask,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
            }

        return (
            oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)),
            kvState,
            currentOffset
        )
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
final class Gemma4TextDecoderLayer: Module {
    let layerType: String
    let enableMoE: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4TextAttention
    @ModuleInfo var mlp: Gemma4TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "router") var router: Gemma4TextRouter?
    @ModuleInfo(key: "experts") var experts: Gemma4TextExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: Gemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.layerType = config.layerTypes[layerIdx]
        self.enableMoE = config.enableMoEBlock
        self._selfAttention.wrappedValue = Gemma4TextAttention(
            config: config, layerIdx: layerIdx, kvSharedOnly: kvSharedOnly)
        self._mlp.wrappedValue = Gemma4TextMLP(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4TextRouter(config: config)
            self._experts.wrappedValue = Gemma4TextExperts(config: config)
            self._postFeedforwardLayerNorm1.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        var residual = x
        var h = inputLayerNorm(x)
        let (attentionOutput, kvState, attentionOffset) = selfAttention(
            h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = attentionOutput
        h = postAttentionLayerNorm(h)
        h = residual + h

        residual = h
        if enableMoE,
            let router,
            let experts,
            let postFeedforwardLayerNorm1,
            let postFeedforwardLayerNorm2,
            let preFeedforwardLayerNorm2
        {
            var dense = preFeedforwardLayerNorm(h)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let (topKIndices, topKWeights) = router(h)
            var sparse = preFeedforwardLayerNorm2(h)
            sparse = experts(sparse, topKIndices: topKIndices, topKWeights: topKWeights)
            sparse = postFeedforwardLayerNorm2(sparse)

            h = dense + sparse
        } else {
            h = preFeedforwardLayerNorm(h)
            h = mlp(h)
        }
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        if let perLayerInputGate, let perLayerProjection, let postPerLayerInputNorm,
            let perLayerInput
        {
            residual = h
            var gated = perLayerInputGate(h)
            gated = geluApproximate(gated)
            gated = gated * perLayerInput
            gated = perLayerProjection(gated)
            gated = postPerLayerInputNorm(gated)
            h = residual + gated
        }

        return (h * layerScalar, kvState, attentionOffset)
    }
}

final class Gemma4TextBackbone: Module, LayerPartitionable, StreamableMoE {
    let config: Gemma4TextConfiguration
    let firstKVSharedLayerIdx: Int
    let layerIdxToCacheIdx: [Int]
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int
    let embedScale: Float
    let embedTokensPerLayerScale: Float
    let perLayerProjectionScale: Float
    private let _perLayerInputScale: MLXArray

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4TextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm:
        Gemma4RMSNormZeroShift?

    // MARK: - LayerPartitionable
    public var gpuLayerCount: Int?
    public var totalLayerCount: Int { layers.count }

    // MARK: - StreamableMoE
    public var streamExperts: Bool = false

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.firstKVSharedLayerIdx = config.hiddenLayers - config.numKVSharedLayers
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        self.embedTokensPerLayerScale = pow(Float(max(config.hiddenSizePerLayerInput, 1)), 0.5)
        self._perLayerInputScale = rsqrt(MLXArray(2.0))

        let concreteLayers = Array(config.layerTypes.prefix(firstKVSharedLayerIdx))
        let sharedFullIdx = concreteLayers.lastIndex(of: "full_attention") ?? 0
        let sharedSlidingIdx = concreteLayers.lastIndex(of: "sliding_attention") ?? 0

        var cacheMap: [Int] = []
        cacheMap.reserveCapacity(config.hiddenLayers)
        for (idx, layerType) in config.layerTypes.enumerated() {
            if idx < firstKVSharedLayerIdx {
                cacheMap.append(idx)
            } else {
                cacheMap.append(layerType == "full_attention" ? sharedFullIdx : sharedSlidingIdx)
            }
        }
        layerIdxToCacheIdx = cacheMap
        firstFullCacheIdx = concreteLayers.firstIndex(of: "full_attention") ?? 0
        firstSlidingCacheIdx = concreteLayers.firstIndex(of: "sliding_attention") ?? 0

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        // KV-shared tail layers (`layer_idx >= hidden_layers - num_kv_shared_layers`)
        // reuse an earlier same-type layer's K/V and own no local K/V projection or
        // K norm. Build them with `kvSharedOnly: true` so the module tree omits those
        // tensors — matching what `sanitize` drops from the checkpoint, and mirroring
        // both `Gemma4AssistantDraftInner` and upstream mlx-lm (`Attention.has_kv`).
        // Without this, shared layers declare a `v_proj`/`k_proj`/`k_norm` that no
        // weight fills, so loading a Gemma 4 checkpoint fails under the strict loader
        // verify with e.g. `keyNotFound(… layers.24.self_attn.v_proj.weight …)`.
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { idx in
            Gemma4TextDecoderLayer(
                config: config, layerIdx: idx,
                kvSharedOnly: firstKVSharedLayer > 0 && idx >= firstKVSharedLayer)
        }
        self._norm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.hiddenSizePerLayerInput > 0 {
            self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabularySizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        } else {
            self.perLayerProjectionScale = 1.0
        }

        super.init()
    }

    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedTokensPerLayer else {
            fatalError("Per-layer inputs requested for a model without embed_tokens_per_layer")
        }
        let validMask =
            logicalAnd(
                inputIds .>= 0, inputIds .< config.vocabularySizePerLayerInput)
        let tokens = MLX.where(validMask, inputIds, MLXArray.zeros(like: inputIds))
        var result = embedTokensPerLayer(tokens)
        result = (result * MLXArray(embedTokensPerLayerScale, dtype: .float32)).asType(result.dtype)
        return result.reshaped(
            Array(inputIds.shape) + [config.hiddenLayers, config.hiddenSizePerLayerInput]
        )
    }

    func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?
    ) -> MLXArray? {
        guard let perLayerModelProjection, let perLayerProjectionNorm else {
            return nil
        }

        var perLayerProjection = perLayerModelProjection(inputsEmbeds) * perLayerProjectionScale
        perLayerProjection = perLayerProjection.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.hiddenLayers, config.hiddenSizePerLayerInput,
            ]
        )
        perLayerProjection = perLayerProjectionNorm(perLayerProjection)

        guard let perLayerInputs else {
            return perLayerProjection
        }

        return (perLayerProjection + perLayerInputs)
            * _perLayerInputScale.asType(inputsEmbeds.dtype)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        emitDrafterState: Bool = false
    ) -> (
        hidden: MLXArray, sharedKV: [String: (MLXArray, MLXArray)]?,
        sharedKVSources: [String: Int]
    ) {
        // Tolerate callers that hand us a 1D `(L,)` token array instead
        // of the canonical 2D `(B, L)` produced by `Gemma4Processor.prepare`.
        // The downstream `perLayerInputs` indexing path (`finalPerLayerInputs[
        // 0..., 0..., idx, 0...]`) requires 4D shapes; with 1D inputs the
        // model otherwise crashes inside `MLXArray.subscript.getter`
        // → `mlx_array_dim` → `_mlx_error`. This expansion is zero-copy
        // and behaves identically when the caller already passed 2D.
        let inputs = inputs.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 }
        let inputsEmbeds = inputsEmbeds.map {
            $0.ndim == 2 ? $0.expandedDimensions(axis: 0) : $0
        }

        let h0: MLXArray
        if let inputsEmbeds {
            h0 = inputsEmbeds
        } else if let inputs {
            let embeddings = embedTokens(inputs)
            h0 = (embeddings * MLXArray(embedScale, dtype: .float32)).asType(embeddings.dtype)
        } else {
            fatalError("Either inputs or inputsEmbeds must be provided")
        }

        let processedPerLayerInputs: MLXArray?
        if config.hiddenSizePerLayerInput > 0 {
            if let perLayerInputs {
                processedPerLayerInputs = perLayerInputs
            } else if let inputs {
                processedPerLayerInputs = getPerLayerInputs(inputs)
            } else {
                processedPerLayerInputs = nil
            }
        } else {
            processedPerLayerInputs = nil
        }
        let finalPerLayerInputs = projectPerLayerInputs(h0, perLayerInputs: processedPerLayerInputs)

        let localCache =
            cache ?? Array(repeating: nil as KVCache?, count: max(firstKVSharedLayerIdx, 1))
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            fullMask = mask
            slidingMask = mask
        } else {
            let tokenTypeIds = tokenTypeIds.map {
                $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0
            }
            let hasAudioTokens =
                if let tokenTypeIds {
                    (tokenTypeIds .== 3).asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let hasVisualTokens =
                if let tokenTypeIds {
                    ((tokenTypeIds .== 1) | (tokenTypeIds .== 2))
                        .asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let useBidirectionalVision =
                config.useBidirectionalAttention == "vision"
                && tokenTypeIds != nil
                && hasVisualTokens
                && !hasAudioTokens
                && h0.dim(1) > 1

            fullMask = createAttentionMask(
                h: h0,
                cache: firstFullCacheIdx < localCache.count ? localCache[firstFullCacheIdx] : nil,
                returnArray: useBidirectionalVision)
            slidingMask = createAttentionMask(
                h: h0,
                cache: firstSlidingCacheIdx < localCache.count
                    ? localCache[firstSlidingCacheIdx] : nil,
                windowSize: config.slidingWindow,
                returnArray: useBidirectionalVision
            )
            if useBidirectionalVision, let tokenTypeIds {
                fullMask = gemma4ApplyBlockwiseBidirectionalOverlay(
                    fullMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: nil
                )
                slidingMask = gemma4ApplyBlockwiseBidirectionalOverlay(
                    slidingMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: config.slidingWindow
                )
            }
        }

        var h = h0
        var intermediates = [(kv: Gemma4SharedKVState?, offset: Int?)](
            repeating: (nil, nil), count: config.hiddenLayers)
        for (idx, layer) in layers.enumerated() {
            let sourceIdx = layerIdxToCacheIdx[idx]
            let layerCache: KVCache? =
                if idx < firstKVSharedLayerIdx, sourceIdx < localCache.count {
                    localCache[sourceIdx]
                } else {
                    nil
                }
            let layerMask =
                if layer.layerType == "full_attention" {
                    fullMask
                } else {
                    slidingMask
                }
            let layerInput: MLXArray? =
                if let finalPerLayerInputs {
                    finalPerLayerInputs[0..., 0..., idx, 0...]
                } else {
                    nil
                }
            let sharedKVForLayer: Gemma4SharedKVState? =
                idx >= firstKVSharedLayerIdx
                ? intermediates[sourceIdx].kv : nil
            let sharedOffsetForLayer: Int? =
                idx >= firstKVSharedLayerIdx
                ? intermediates[sourceIdx].offset : nil
            let (output, kvState, attentionOffset) = partitionedLayerCall(
                index: idx, gpuLayerCount: gpuLayerCount, stream: streamExperts
            ) {
                layer(
                    h,
                    mask: layerMask,
                    cache: layerCache,
                    perLayerInput: layerInput,
                    sharedKV: sharedKVForLayer,
                    offset: sharedOffsetForLayer
                )
            }
            h = output
            intermediates[idx] = (kvState, attentionOffset)
        }
        let finalHidden = norm(h)

        guard emitDrafterState else {
            return (finalHidden, nil, [:])
        }

        // Walk intermediates from the last layer backward; for each unique
        // `layer_type`, take the first `.regular` K/V encountered. Quantized
        // cases are skipped — the iterator treats absent `sharedKV` as a
        // signal to fall back to single-token generation (R8/R13 limitation,
        // documented).
        var sharedKV: [String: (MLXArray, MLXArray)] = [:]
        // Which cache entry each emitted tuple came from. The consumer reconciles the emitted
        // snapshot against the cache after a speculative commit, and it can only do that exactly
        // if it knows the entry -- a sliding layer's snapshot is bounded by its ring, a global
        // layer's is not, and the two are indistinguishable by length at the crossing.
        var sharedKVSources: [String: Int] = [:]
        var seenTypes = Set<String>()
        let targetTypes: Set<String> = ["full_attention", "sliding_attention"]
        for idx in stride(from: layers.count - 1, through: 0, by: -1) {
            let layerType = layers[idx].layerType
            guard targetTypes.contains(layerType), !seenTypes.contains(layerType) else {
                continue
            }
            if case .regular(let keys, let values) = intermediates[idx].kv {
                sharedKV[layerType] = (keys, values)
                // Recorded here rather than derived from `config.layerTypes`: the walk keeps
                // descending past a quantized entry, so which layer supplies a type is a runtime
                // fact.
                sharedKVSources[layerType] = layerIdxToCacheIdx[idx]
                seenTypes.insert(layerType)
            }
            if seenTypes == targetTypes { break }
        }
        // Treat partial coverage (e.g. only one layer_type populated, or
        // quantized cache for the other) as no-emit — iterator falls back.
        let complete = seenTypes == targetTypes
        return (finalHidden, complete ? sharedKV : nil, complete ? sharedKVSources : [:])
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift` (the MTP drafter
/// reaches `embed_tokens` / `embed_scale` / `config.layer_types` through this).
final class Gemma4TextLanguageModel: Module, KVCacheDimensionProvider {
    let config: Gemma4TextConfiguration
    let finalLogitSoftcapping: Float?

    @ModuleInfo(key: "model") var model: Gemma4TextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    var kvHeads: [Int] {
        (0 ..< config.hiddenLayers).map { idx in
            let layerType = config.layerTypes[idx]
            if config.attentionKEqV && layerType == "full_attention" {
                return config.globalKVHeads ?? config.kvHeads
            } else {
                return config.kvHeads
            }
        }
    }

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._model.wrappedValue = Gemma4TextBackbone(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        return try config.layerTypes.prefix(config.hiddenLayers - config.numKVSharedLayers).map {
            layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: slidingWindow,
                usesSlidingWindow: layerType != "full_attention")
        }
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        emitDrafterState: Bool = false,
        lastPositionOnly: Bool = false
    ) -> LMOutput {
        let (hidden, sharedKV, sharedKVSources) = model(
            inputs, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache?.map { $0 as KVCache? },
            perLayerInputs: perLayerInputs,
            tokenTypeIds: tokenTypeIds,
            emitDrafterState: emitDrafterState
        )
        let projectionInput: MLXArray
        if lastPositionOnly, hidden.dim(1) > 1 {
            projectionInput = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        } else {
            projectionInput = hidden
        }
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(projectionInput)
        } else {
            logits = model.embedTokens.asLinear(projectionInput)
        }
        let softcappedLogits: MLXArray
        if let finalLogitSoftcapping, finalLogitSoftcapping > 0 {
            let scale = MLXArray(finalLogitSoftcapping)
            softcappedLogits = gemma4LogitSoftcap(logits, scale)
        } else {
            softcappedLogits = logits
        }

        guard emitDrafterState, let sharedKV else {
            return LMOutput(logits: softcappedLogits)
        }
        var state = LMOutput.State()
        state[mtpLastHiddenStatesKey] = hidden
        state[mtpSharedKVStatesKey] = sharedKV
        state[mtpSharedKVSourceIndicesKey] = sharedKVSources
        return LMOutput(logits: softcappedLogits, state: state)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count + 1)

        for (key, value) in weights {
            if key.contains("rotary_emb") {
                continue
            }
            // Drop redundant k_proj/v_proj/k_norm for KV-shared layers: they reuse an
            // earlier layer's K/V and own no K projection or K norm, so the module tree
            // has none. QAT checkpoints already omit these; some (PTQ) checkpoints still
            // ship them, and keeping them would be an unexpected weight.
            // Scope: text backbone only — the vision/audio towers share the
            // `layers.N.self_attn.{k,v}_proj` naming, so without these guards the drop
            // would amputate tower layers >= firstKVSharedLayer.
            if firstKVSharedLayer > 0,
                !key.contains("vision_tower"),
                !key.contains("audio_tower"),
                key.contains("self_attn.k_proj")
                    || key.contains("self_attn.v_proj")
                    || key.contains("self_attn.k_norm"),
                let layerIdx = Self.decoderLayerIndex(in: key),
                layerIdx >= firstKVSharedLayer
            {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            }
            if newKey.hasPrefix("language_model."),
                !newKey.hasPrefix("language_model.model."),
                !newKey.hasPrefix("language_model.lm_head.")
            {
                let rest = String(newKey.dropFirst("language_model.".count))
                newKey = "language_model.model.\(rest)"
            }

            if newKey.hasSuffix(".experts.down_proj") {
                newKey = newKey.replacingOccurrences(
                    of: ".experts.down_proj",
                    with: ".experts.switch_glu.down_proj.weight"
                )
            }

            if newKey.hasSuffix(".experts.gate_up_proj") {
                let mid = value.dim(-2) / 2
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.gate_proj.weight"
                    )
                ] = value[.ellipsis, ..<mid, 0...]
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.up_proj.weight"
                    )
                ] = value[.ellipsis, mid..., 0...]
                continue
            }

            sanitized[newKey] = value
        }

        if config.tieWordEmbeddings {
            sanitized = sanitized.filter { key, _ in
                !key.hasPrefix("language_model.lm_head.")
            }
        } else if sanitized["language_model.lm_head.weight"] == nil,
            let embedWeight = sanitized["language_model.model.embed_tokens.weight"]
        {
            sanitized["language_model.lm_head.weight"] = embedWeight
        }

        return sanitized
    }

    /// Extract `N` from a weight key shaped like `…layers.N.…`, else nil.
    private static func decoderLayerIndex(in key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        let digits = key[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }
}

// Shared encoders keep the existing MLXVLM names available.
public typealias Gemma4VisionConfiguration = MLXLMCommon.Gemma4VisionConfiguration
package typealias Gemma4VisionModel = MLXLMCommon.Gemma4VisionModel
package typealias Gemma4MultimodalEmbedder = MLXLMCommon.Gemma4MultimodalEmbedder

// MARK: - Model

public final class Gemma4: Module, VLMModel, KVCacheDimensionProvider, StreamableMoE,
    SuppressedTokensProviding
{
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    /// Module-internal — also reached by `Gemma4Assistant.swift` (drafter `bind()`
    /// walks here to cache the target's input embeddings, embed scale, and
    /// per-layer type metadata).
    @ModuleInfo(key: "language_model") var languageModel: Gemma4TextLanguageModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder

    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma4AudioModel?
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma4MultimodalEmbedder?

    public let config: Gemma4Configuration
    public var suppressedTokenIds: Set<Int>

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    // MARK: - StreamableMoE (delegates to the backbone)
    public var streamExperts: Bool {
        get { languageModel.model.streamExperts }
        set { languageModel.model.streamExperts = newValue }
    }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        self.suppressedTokenIds = Set(
            [
                config.imageTokenId, config.audioTokenId, config.videoTokenId,
                config.boiTokenId, config.eoiTokenId,
            ].compactMap { $0 })
        self._visionTower.wrappedValue = Gemma4VisionModel(config: config.visionConfiguration)
        self._languageModel.wrappedValue = Gemma4TextLanguageModel(config.textConfiguration)
        self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
            embeddingDim: config.visionConfiguration.hiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            eps: config.visionConfiguration.rmsNormEps
        )
        if let acfg = config.audioConfig {
            self._audioTower.wrappedValue = Gemma4AudioModel(config: acfg)
            self._embedAudio.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: acfg.outputProjDims,
                textHiddenSize: config.textConfiguration.hiddenSize,
                eps: config.visionConfiguration.rmsNormEps
            )
        }
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil,
        audioValues: MLXArray? = nil,
        audioMask: MLXArray? = nil
    ) throws -> (MLXArray, MLXArray?) {
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)
        inputsEmbeds =
            (inputsEmbeds
            * MLXArray(pow(Float(config.textConfiguration.hiddenSize), 0.5), dtype: .float32))
            .asType(inputsEmbeds.dtype)

        var perLayerInputs: MLXArray? = nil
        if config.textConfiguration.hiddenSizePerLayerInput > 0 {
            // Per-layer inputs are text-only: zero out every multimodal soft token
            // (image / audio / video) so their ids don't index the PLE embedding.
            var multimodalMask = inputIds .== config.imageTokenId
            if let audioTokenId = config.audioTokenId {
                multimodalMask = multimodalMask | (inputIds .== audioTokenId)
            }
            if let videoTokenId = config.videoTokenId {
                multimodalMask = multimodalMask | (inputIds .== videoTokenId)
            }
            let perLayerTokens = MLX.where(
                logicalNot(multimodalMask), inputIds, MLXArray.zeros(like: inputIds))
            let layerInputs = languageModel.model.getPerLayerInputs(perLayerTokens)
            let mask = multimodalMask.reshaped(multimodalMask.dim(0), multimodalMask.dim(1), 1, 1)
            perLayerInputs = MLX.where(mask, MLXArray.zeros(like: layerInputs), layerInputs)
        }

        if let image {
            // Images keep their own aspect-preserving sizes: the processor
            // zero-pads them onto a shared canvas and records each real size in
            // frames. Slice each image back out, run the tower on it alone, and
            // concatenate the pooled tokens in placeholder order.
            let pixels =
                if image.pixels.ndim == 3 {
                    expandedDimensions(image.pixels, axis: 0)
                } else {
                    image.pixels
                }
            let frames =
                image.frames
                ?? Array(repeating: THW(1, pixels.dim(2), pixels.dim(3)), count: pixels.dim(0))
            var perImageFeatures: [MLXArray] = []
            for (index, frame) in frames.enumerated() {
                let imagePixels = pixels[index ..< index + 1, 0..., ..<frame.h, ..<frame.w]
                perImageFeatures.append(visionTower(imagePixels))
            }
            var imageFeatures =
                perImageFeatures.count == 1
                ? perImageFeatures[0]
                : concatenated(perImageFeatures, axis: 1)
            imageFeatures = embedVision(imageFeatures)
            imageFeatures = imageFeatures.asType(inputsEmbeds.dtype)

            let imageMask = inputIds .== config.imageTokenId
            let expectedImageTokens = imageMask.asType(.int32).sum().item(Int.self)

            if expectedImageTokens != imageFeatures.dim(1) {
                throw Gemma4Error.multimodalTokenCountMismatch(
                    kind: "image", featureTokens: imageFeatures.dim(1),
                    promptTokens: expectedImageTokens)
            }

            var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
            imageMaskExpanded = broadcast(imageMaskExpanded, to: inputsEmbeds.shape)
            inputsEmbeds = gemma4MaskedScatter(
                inputTensor: inputsEmbeds,
                mask: imageMaskExpanded,
                source: imageFeatures
            )
        }

        if let video {
            guard let videoTokenId = config.videoTokenId else {
                throw Gemma4Error.missingVideoTokenId
            }
            inputsEmbeds = try scatterVideoFeatures(
                into: inputsEmbeds, inputIds: inputIds, video: video, tokenId: videoTokenId)
        }

        if let audioValues = audioValues, let audioTower = audioTower, let embedAudio = embedAudio,
            let audioTokenId = config.audioTokenId
        {
            let actualAudioMask =
                audioMask ?? MLXArray.ones(audioValues.shape[0 ..< 2], dtype: .bool)
            print(
                "[Omni Audio] audioValues=\(audioValues.shape) mask=\(actualAudioMask.shape) maskSum=\(actualAudioMask.asType(.int32).sum().item(Int.self))"
            )
            let (audioOutputs, _) = audioTower(audioValues, mask: actualAudioMask)
            eval(audioOutputs)
            let audioFeatures = embedAudio(audioOutputs).asType(inputsEmbeds.dtype)
            eval(audioFeatures)

            let audioTokenMask = inputIds .== audioTokenId
            let audioTokenCount = audioTokenMask.asType(.int32).sum().item(Int.self)
            let audioFeatureCount = audioFeatures.dim(1)
            let audioStd = MLX.sqrt(MLX.variance(audioOutputs)).item(Float.self)
            let audioAbsMean = MLX.abs(audioOutputs).mean().item(Float.self)
            print(
                "[Omni Audio] audioOutputs=\(audioOutputs.shape) mean=\(audioOutputs.mean().item(Float.self)) std=\(audioStd) absMean=\(audioAbsMean) tokenCount=\(audioTokenCount) featureCount=\(audioFeatureCount)"
            )
            guard audioTokenCount == audioFeatureCount else {
                print(
                    "[Gemma4] Omni audio token count mismatch: \(audioTokenCount) tokens vs \(audioFeatureCount) features. Skipping."
                )
                return (inputsEmbeds, perLayerInputs)
            }

            // Sample embed value at first audio token position BEFORE scatter
            let firstAudioPos = MLX.argMax(audioTokenMask[0].asType(.int32), axis: 0).item(Int.self)
            let embedBefore = inputsEmbeds[0, firstAudioPos, 0].item(Float.self)

            var audioMaskExpanded = expandedDimensions(audioTokenMask, axis: -1)
            audioMaskExpanded = broadcast(audioMaskExpanded, to: inputsEmbeds.shape)
            inputsEmbeds = gemma4MaskedScatter(
                inputTensor: inputsEmbeds,
                mask: audioMaskExpanded,
                source: audioFeatures
            )
            eval(inputsEmbeds)
            let embedAfter = inputsEmbeds[0, firstAudioPos, 0].item(Float.self)
            print(
                "[Omni Audio] scatter: pos=\(firstAudioPos) before=\(embedBefore) after=\(embedAfter) changed=\(embedBefore != embedAfter)"
            )
        }

        return (inputsEmbeds, perLayerInputs)
    }

    private func scatterVideoFeatures(
        into inputsEmbeds: MLXArray, inputIds: MLXArray, video: LMInput.ProcessedVideo,
        tokenId: Int
    ) throws -> MLXArray {
        let videoPixelValues = video.pixels
        guard videoPixelValues.ndim == 4, videoPixelValues.dim(0) > 0 else {
            throw Gemma4Error.emptyVideoFrames
        }
        guard config.visionSoftTokensPerVideoFrame > 0, config.videoFrameChunkSize > 0 else {
            throw VLMError.processing(
                "Gemma4 video requires positive token counts and chunk sizes.")
        }
        let frames =
            video.frames
            ?? Array(
                repeating: THW(1, videoPixelValues.dim(2), videoPixelValues.dim(3)),
                count: videoPixelValues.dim(0))
        guard frames.count == videoPixelValues.dim(0) else {
            throw VLMError.processing("Gemma4 video requires one geometry record per frame.")
        }
        let poolingSide =
            config.visionConfiguration.patchSize
            * config.visionConfiguration.poolingKernelSize
        var requestedTokens = 0
        for frame in frames {
            guard frame.t == 1, frame.h > 0, frame.w > 0,
                frame.h <= videoPixelValues.dim(2), frame.w <= videoPixelValues.dim(3),
                poolingSide > 0, frame.h % poolingSide == 0, frame.w % poolingSide == 0
            else {
                throw VLMError.processing(
                    "Gemma4 video frame dimensions must fit the pixel canvas and divide into complete pooling cells."
                )
            }
            let count = (frame.h / poolingSide) * (frame.w / poolingSide)
            guard count <= config.visionSoftTokensPerVideoFrame else {
                throw VLMError.processing(
                    "Gemma4 video frame produces \(count) features, exceeding its \(config.visionSoftTokensPerVideoFrame)-token maximum. Resize the frame before encoding."
                )
            }
            requestedTokens += count
        }
        let expectedTokens = (inputIds .== tokenId).asType(.int32).sum().item(Int.self)
        guard expectedTokens == requestedTokens else {
            throw Gemma4Error.multimodalTokenCountMismatch(
                kind: "video", featureTokens: requestedTokens, promptTokens: expectedTokens)
        }
        // Chunking limits vision intermediates. Retained features and text context grow with frames.
        var chunks: [MLXArray] = []
        let chunkSize = config.videoFrameChunkSize
        var start = 0
        while start < frames.count {
            let frame = frames[start]
            var end = start + 1
            while end < min(start + chunkSize, frames.count),
                frames[end].h == frame.h, frames[end].w == frame.w
            {
                end += 1
            }
            var features = visionTower(
                videoPixelValues[start ..< end, 0..., ..<frame.h, ..<frame.w])
            features = embedVision(features)
            features = features.reshaped(1, -1, features.dim(-1)).asType(inputsEmbeds.dtype)
            eval(features)
            chunks.append(features)
            start = end
        }
        let features = concatenated(chunks, axis: 1)
        let producedTokens = features.dim(1)
        guard expectedTokens == producedTokens else {
            throw Gemma4Error.multimodalTokenCountMismatch(
                kind: "video", featureTokens: producedTokens, promptTokens: expectedTokens)
        }
        let mask = broadcast((inputIds .== tokenId)[.ellipsis, .newAxis], to: inputsEmbeds.shape)
        return gemma4MaskedScatter(inputTensor: inputsEmbeds, mask: mask, source: features)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        if input.image == nil, input.video == nil, input.audio == nil {
            return try gemma4PrepareTextOnly(
                input, cache: cache, prefill: prefill, languageModel: languageModel)
        }
        let (embeddings, perLayerInputs) = try getInputEmbeddings(
            inputIds: input.text.tokens, image: input.image, video: input.video,
            audioValues: input.audio?.features, audioMask: input.audio?.mask)
        let result = languageModel(
            nil, cache: cache, inputsEmbeds: embeddings, perLayerInputs: perLayerInputs,
            tokenTypeIds: gemma4TokenTypeIds(
                inputIds: input.text.tokens, imageTokenId: config.imageTokenId,
                videoTokenId: config.videoTokenId, audioTokenId: config.audioTokenId),
            emitDrafterState: state?[mtpEmitFlagKey] ?? false,
            lastPositionOnly: true)
        prefill.progress?(embeddings.dim(1), embeddings.dim(1))
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let logits = languageModel(inputs, cache: cache?.map { $0 })
        return logits.logits
    }

    /// MTP-aware `LanguageModel` entry point. Reads `mtpEmitFlagKey` from
    /// the incoming `state` and threads it through to `Gemma4TextLanguageModel`;
    /// the returned `LMOutput` carries `mtpLastHiddenStatesKey` and
    /// `mtpSharedKVStatesKey` populated when the flag is set, empty otherwise.
    /// Overrides the protocol-extension default at `LanguageModel` which
    /// would discard `state`.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let emit = state?[mtpEmitFlagKey] ?? false
        return languageModel(
            input.tokens, cache: cache?.map { $0 },
            emitDrafterState: emit
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = languageModel.sanitize(weights: weights)

        if !config.visionConfiguration.useClippedLinears {
            sanitized = sanitized.filter { key, _ in
                !key.contains("input_min")
                    && !key.contains("input_max")
                    && !key.contains("output_min")
                    && !key.contains("output_max")
            }
        }

        var finalSanitized: [String: MLXArray] = [:]
        for (key, value) in sanitized {
            var newKey = key
            if newKey.contains("subsampling_conv.layers.") {
                newKey = newKey.replacingOccurrences(
                    of: "subsampling_conv.layers.0", with: "subsample_conv_projection.layer0")
                newKey = newKey.replacingOccurrences(
                    of: "subsampling_conv.layers.1", with: "subsample_conv_projection.layer1")
            }
            finalSanitized[newKey] = value
        }

        return finalSanitized
    }
}

// MARK: - Gemma 4 Unified

public struct Gemma4UnifiedAudioConfiguration: Codable, Sendable {
    public let modelType: String
    public let audioSamplesPerToken: Int
    public let audioEmbedDim: Int
    public let hiddenSize: Int
    public let outputProjectionDimensions: Int
    public let rmsNormEps: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioSamplesPerToken = "audio_samples_per_token"
        case audioEmbedDim = "audio_embed_dim"
        case hiddenSize = "hidden_size"
        case outputProjectionDimensions = "output_proj_dims"
        case rmsNormEps = "rms_norm_eps"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType)
            ?? "gemma4_unified_audio"
        audioSamplesPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSamplesPerToken) ?? 640
        audioEmbedDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioEmbedDim) ?? 640
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize) ?? 640
        outputProjectionDimensions =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.outputProjectionDimensions) ?? 640
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
    }
}

public struct Gemma4UnifiedVisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let modelPatchSize: Int
    public let mmEmbedDim: Int
    public let mmPositionEmbeddingSize: Int
    public let numSoftTokens: Int
    public let rmsNormEps: Float
    public let outputProjectionDimensions: Int

    public var hiddenSize: Int { outputProjectionDimensions }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case modelPatchSize = "model_patch_size"
        case mmEmbedDim = "mm_embed_dim"
        case mmPositionEmbeddingSize = "mm_posemb_size"
        case numSoftTokens = "num_soft_tokens"
        case rmsNormEps = "rms_norm_eps"
        case outputProjectionDimensions = "output_proj_dims"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType)
            ?? "gemma4_unified_vision"
        patchSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize) ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize) ?? 3
        modelPatchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.modelPatchSize)
            ?? patchSize * poolingKernelSize
        mmEmbedDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.mmEmbedDim) ?? 3840
        mmPositionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.mmPositionEmbeddingSize) ?? 1120
        numSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.numSoftTokens) ?? 280
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        outputProjectionDimensions =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.outputProjectionDimensions) ?? 3840
    }
}

public struct Gemma4UnifiedConfiguration: Decodable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4UnifiedVisionConfiguration?
    public let audioConfiguration: Gemma4UnifiedAudioConfiguration?
    public let modelType: String
    public let quantization: BaseConfiguration.Quantization?
    public let imageTokenId: Int
    public let audioTokenId: Int?
    public let videoTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let boaTokenId: Int
    public let eoaTokenId: Int?
    public let visionSoftTokensPerImage: Int
    public let visionSoftTokensPerVideoFrame: Int
    public let audioSoftTokensPerImage: Int
    public let audioMsPerToken: Int
    public let tieWordEmbeddings: Bool

    private let _vocabularySize: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int { _vocabularySize ?? textConfiguration.vocabularySize }
    public var hiddenSize: Int { _hiddenSize ?? textConfiguration.hiddenSize }
    public var padTokenId: Int { _padTokenId ?? 0 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case audioConfiguration = "audio_config"
        case modelType = "model_type"
        case quantization
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
        case eoaTokenIndex = "eoa_token_index"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case visionSoftTokensPerVideoFrame = "vision_soft_tokens_per_video_frame"
        case audioSoftTokensPerImage = "audio_soft_tokens_per_image"
        case audioMsPerToken = "audio_ms_per_token"
        case tieWordEmbeddings = "tie_word_embeddings"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration =
            try c.decodeIfPresent(
                Gemma4TextConfiguration.self, forKey: CodingKeys.textConfiguration)
            ?? Gemma4TextConfiguration(from: decoder)
        visionConfiguration = try c.decodeIfPresent(
            Gemma4UnifiedVisionConfiguration.self, forKey: CodingKeys.visionConfiguration)
        audioConfiguration = try c.decodeIfPresent(
            Gemma4UnifiedAudioConfiguration.self, forKey: CodingKeys.audioConfiguration)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_unified"
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: CodingKeys.quantization)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId) ?? 258_881
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId) ?? 258_884
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
        boaTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boaTokenId) ?? 256_000
        eoaTokenId =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoaTokenId)
            ?? c.decodeIfPresent(Int.self, forKey: CodingKeys.eoaTokenIndex)
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerImage) ?? 280
        visionSoftTokensPerVideoFrame =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerVideoFrame) ?? 70
        audioSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSoftTokensPerImage) ?? 750
        audioMsPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioMsPerToken) ?? 40
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings) ?? true
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._vocabularySize)
        _hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._hiddenSize)
        _padTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys._padTokenId)
    }
}

private final class Gemma4UnifiedVisionEmbedder: Module {
    let patchDim: Int

    @ModuleInfo(key: "patch_ln1") var patchLayerNorm1: LayerNorm
    @ModuleInfo(key: "patch_dense") var patchDense: Linear
    @ModuleInfo(key: "patch_ln2") var patchLayerNorm2: LayerNorm
    @ParameterInfo(key: "pos_embedding") var positionEmbedding: MLXArray
    @ModuleInfo(key: "pos_norm") var positionNorm: LayerNorm

    init(config: Gemma4UnifiedVisionConfiguration) {
        self.patchDim = config.modelPatchSize * config.modelPatchSize * 3
        self._patchLayerNorm1.wrappedValue = LayerNorm(dimensions: patchDim)
        self._patchDense.wrappedValue = Linear(patchDim, config.mmEmbedDim)
        self._patchLayerNorm2.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim)
        self._positionEmbedding.wrappedValue = MLXArray.zeros([
            config.mmPositionEmbeddingSize, 2, config.mmEmbedDim,
        ])
        self._positionNorm.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim)
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray, imagePositionIds: MLXArray? = nil) -> MLXArray {
        var pixels = pixelValues
        if pixels.ndim == 4 && pixels.dim(-1) == patchDim {
            pixels = pixels.reshaped(pixels.dim(0), -1, patchDim)
        }

        var hiddenStates = patchLayerNorm1(pixels)
        hiddenStates = patchDense(hiddenStates)
        hiddenStates = patchLayerNorm2(hiddenStates)

        if let imagePositionIds {
            let clamped = maximum(imagePositionIds, MLXArray(0)).asType(.int32)
            let batch = clamped.dim(0)
            let sequenceLength = clamped.dim(1)
            let xIndices = clamped[0..., 0..., 0].flattened()
            let yIndices = clamped[0..., 0..., 1].flattened()
            let xEmbeddings = take(positionEmbedding[0..., 0], xIndices, axis: 0)
                .reshaped(batch, sequenceLength, -1)
            let yEmbeddings = take(positionEmbedding[0..., 1], yIndices, axis: 0)
                .reshaped(batch, sequenceLength, -1)
            let valid = (imagePositionIds .!= -1).asType(hiddenStates.dtype)
            hiddenStates =
                hiddenStates
                + xEmbeddings * expandedDimensions(valid[0..., 0..., 0], axis: -1)
                + yEmbeddings * expandedDimensions(valid[0..., 0..., 1], axis: -1)
        }

        return positionNorm(hiddenStates)
    }
}

public final class Gemma4Unified: Module, VLMModel, KVCacheDimensionProvider,
    SuppressedTokensProviding
{
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4TextLanguageModel
    @ModuleInfo(key: "vision_embedder") private var visionEmbedder: Gemma4UnifiedVisionEmbedder?
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder?
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma4MultimodalEmbedder?

    public let config: Gemma4UnifiedConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    /// Multimodal placeholder token IDs that must never be sampled as text.
    /// Seeded from the model configuration; the model factory merges
    /// `suppress_tokens` from `generation_config.json` on top.
    /// See `SuppressedTokensProviding` in MLXLMCommon.
    public var suppressedTokenIds: Set<Int>

    public init(_ config: Gemma4UnifiedConfiguration) {
        self.config = config
        self.suppressedTokenIds = Set(
            [
                config.imageTokenId, config.audioTokenId, config.videoTokenId,
                config.boiTokenId, config.eoiTokenId, config.boaTokenId, config.eoaTokenId,
            ].compactMap { $0 })
        self._languageModel.wrappedValue = Gemma4TextLanguageModel(config.textConfiguration)
        if let visionConfiguration = config.visionConfiguration {
            self._visionEmbedder.wrappedValue = Gemma4UnifiedVisionEmbedder(
                config: visionConfiguration)
            self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: visionConfiguration.outputProjectionDimensions,
                textHiddenSize: config.textConfiguration.hiddenSize,
                eps: visionConfiguration.rmsNormEps
            )
        }
        if let audioConfiguration = config.audioConfiguration {
            self._embedAudio.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: audioConfiguration.outputProjectionDimensions,
                textHiddenSize: config.textConfiguration.hiddenSize,
                eps: audioConfiguration.rmsNormEps
            )
        }
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    private func getImageFeatures(
        pixelValues: MLXArray,
        imagePositionIds: MLXArray?
    ) throws -> MLXArray {
        guard let visionEmbedder, let embedVision else {
            throw VLMError.processing("Vision inputs were provided, but vision_config is missing.")
        }
        var projected = embedVision(visionEmbedder(pixelValues, imagePositionIds: imagePositionIds))
        if let imagePositionIds {
            let validMask =
                (imagePositionIds[0..., 0..., 0] .!= -1)
                | (imagePositionIds[0..., 0..., 1] .!= -1)
            projected = gemma4CompactPrefixRows(features: projected, validMask: validMask)
        } else {
            projected = projected.reshaped(-1, projected.dim(-1))
        }
        return projected
    }

    private func getAudioFeatures(features: MLXArray, mask: MLXArray?) throws -> MLXArray {
        guard let embedAudio else {
            throw VLMError.processing("Audio inputs were provided, but audio_config is missing.")
        }
        var projected = embedAudio(features)
        if let mask {
            projected = gemma4CompactPrefixRows(features: projected, validMask: mask)
        } else {
            projected = projected.reshaped(-1, projected.dim(-1))
        }
        return projected
    }

    private func scatterFeatures(
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        tokenId: Int?,
        features: MLXArray?,
        kind: String
    ) throws -> MLXArray {
        guard let tokenId, let features else {
            return inputEmbeds
        }
        let tokenMask = inputIds .== tokenId
        let expectedTokens = tokenMask.asType(.int32).sum().item(Int.self)
        guard expectedTokens == features.dim(0) else {
            throw Gemma4Error.multimodalTokenCountMismatch(
                kind: kind, featureTokens: features.dim(0), promptTokens: expectedTokens)
        }
        var expandedMask = expandedDimensions(tokenMask, axis: -1)
        expandedMask = broadcast(expandedMask, to: inputEmbeds.shape)
        return gemma4MaskedScatter(
            inputTensor: inputEmbeds,
            mask: expandedMask,
            source: features.asType(inputEmbeds.dtype)
        )
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        imagePositionIds: MLXArray? = nil,
        videoPixelValues: MLXArray? = nil,
        videoPositionIds: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioMask: MLXArray? = nil
    ) throws -> (MLXArray, MLXArray?) {
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)
        inputsEmbeds =
            (inputsEmbeds
            * MLXArray(pow(Float(config.textConfiguration.hiddenSize), 0.5), dtype: .float32))
            .asType(inputsEmbeds.dtype)

        var perLayerInputs: MLXArray? = nil
        if config.textConfiguration.hiddenSizePerLayerInput > 0 {
            var multimodalMask = inputIds .== config.imageTokenId
            if let audioTokenId = config.audioTokenId {
                multimodalMask = multimodalMask | (inputIds .== audioTokenId)
            }
            if let videoTokenId = config.videoTokenId {
                multimodalMask = multimodalMask | (inputIds .== videoTokenId)
            }
            let perLayerTokens = MLX.where(
                logicalNot(multimodalMask), inputIds, MLXArray.zeros(like: inputIds))
            perLayerInputs = languageModel.model.getPerLayerInputs(perLayerTokens)
        }

        let imageFeatures =
            if let pixelValues {
                try getImageFeatures(pixelValues: pixelValues, imagePositionIds: imagePositionIds)
            } else {
                nil as MLXArray?
            }
        inputsEmbeds = try scatterFeatures(
            inputEmbeds: inputsEmbeds,
            inputIds: inputIds,
            tokenId: config.imageTokenId,
            features: imageFeatures,
            kind: "image"
        )

        let videoFeatures =
            if let videoPixelValues {
                try getImageFeatures(
                    pixelValues: videoPixelValues, imagePositionIds: videoPositionIds)
            } else {
                nil as MLXArray?
            }
        inputsEmbeds = try scatterFeatures(
            inputEmbeds: inputsEmbeds,
            inputIds: inputIds,
            tokenId: config.videoTokenId,
            features: videoFeatures,
            kind: "video"
        )

        let audioFeatures =
            if let audioFeatures {
                try getAudioFeatures(features: audioFeatures, mask: audioMask)
            } else {
                nil as MLXArray?
            }
        inputsEmbeds = try scatterFeatures(
            inputEmbeds: inputsEmbeds,
            inputIds: inputIds,
            tokenId: config.audioTokenId,
            features: audioFeatures,
            kind: "audio"
        )

        return (inputsEmbeds, perLayerInputs)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        if input.image == nil, input.video == nil, input.audio == nil {
            return try gemma4PrepareTextOnly(
                input, cache: cache, prefill: prefill, languageModel: languageModel)
        }

        let (inputsEmbeds, perLayerInputs) = try getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: input.image?.pixels,
            imagePositionIds: input.image?.positionIds,
            videoPixelValues: input.video?.pixels,
            videoPositionIds: input.video?.positionIds,
            audioFeatures: input.audio?.features,
            audioMask: input.audio?.mask
        )
        let tokenTypeIds = gemma4TokenTypeIds(
            inputIds: input.text.tokens,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            audioTokenId: config.audioTokenId
        )
        let result = languageModel(
            nil,
            cache: cache.map { $0 },
            inputsEmbeds: inputsEmbeds,
            perLayerInputs: perLayerInputs,
            tokenTypeIds: tokenTypeIds,
            lastPositionOnly: true
        )
        let total = inputsEmbeds.dim(1)
        prefill.progress?(total, total)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let logits = languageModel(inputs, cache: cache?.map { $0 })
        return logits.logits
    }

    /// MTP-aware `LanguageModel` entry point. Reads `mtpEmitFlagKey` from
    /// the incoming `state` and threads it through to `Gemma4TextLanguageModel`;
    /// the returned `LMOutput` carries `mtpLastHiddenStatesKey` and
    /// `mtpSharedKVStatesKey` populated when the flag is set, empty otherwise.
    /// Overrides the protocol-extension default at `LanguageModel` which
    /// would discard `state`. Mirrors `Gemma4.callAsFunction(_:cache:state:)`.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let emit = state?[mtpEmitFlagKey] ?? false
        return languageModel(
            input.tokens, cache: cache?.map { $0 },
            emitDrafterState: emit
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var filtered: [String: MLXArray] = [:]
        filtered.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key == "lm_head.weight" {
                continue
            }
            if embedAudio == nil && key.contains("embed_audio") {
                continue
            }
            if visionEmbedder == nil
                && (key.contains("vision_embedder") || key.contains("embed_vision"))
            {
                continue
            }
            filtered[key] = value
        }
        return languageModel.sanitize(weights: filtered)
    }
}

protocol Gemma4BackboneProviding {
    var textBackbone: Gemma4TextBackbone { get }
}

extension Gemma4: Gemma4BackboneProviding {
    var textBackbone: Gemma4TextBackbone { languageModel.model }
}

extension Gemma4Unified: Gemma4BackboneProviding {
    var textBackbone: Gemma4TextBackbone { languageModel.model }
}

// MARK: - Message Generator

/// Message generator for Gemma4/Gemma3n that places images BEFORE text in the content
/// array, matching the Python reference `apply_chat_template(image_first=True)` behaviour.
/// The Gemma4 Jinja template processes content in array order, so ordering matters:
///   Python:  [image, text, audio]  →  <|image|>\ntext<|audio|>
///   (Wrong): [text, image, audio]  →  text<|image|><|audio|>  (model ignores audio)
public struct Gemma4MessageGenerator: MessageGenerator {
    private let videoPlaceholder: String

    public init(videoPlaceholder: String = Gemma4Processor.videoPlaceholder) {
        self.videoPlaceholder = videoPlaceholder
    }

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        // System/assistant/tool roles: no media, use plain string content
        if message.role != .user
            || (message.images.isEmpty && message.audio.isEmpty && message.videos.isEmpty)
        {
            var dict: [String: any Sendable] = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            addToolMetadata(to: &dict, for: message)
            return dict
        }
        // User messages with media: images FIRST, then text, then audio
        // This matches the Python: apply_chat_template(..., image_first=True)
        var content: [[String: any Sendable]] = []
        content += message.images.map { _ in ["type": "image"] as [String: any Sendable] }
        let placeholders = String(repeating: videoPlaceholder, count: message.videos.count)
        content.append(["type": "text", "text": placeholders + message.content])
        content += message.audio.map { _ in ["type": "audio"] as [String: any Sendable] }
        var result: MLXLMCommon.Message = ["role": message.role.rawValue, "content": content]
        addToolMetadata(to: &result, for: message)
        return result
    }
}

// MARK: - Processor

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer
    public static let videoPlaceholder = "<|video|>"

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let processedImage = MediaProcessing.apply(image, processing: processing)
        let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
        let targetSize = config.aspectPreservingTargetSize(for: srgbImage.extent.size)
        let resizedImage =
            srgbImage.extent.size == targetSize
            ? srgbImage
            : MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
        let finalImage =
            if config.doNormalize {
                MediaProcessing.normalize(
                    resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
            } else {
                resizedImage
            }
        let pixelValues = MediaProcessing.asMLXArray(finalImage)

        return (pixelValues, THW(1, Int(targetSize.height), Int(targetSize.width)))
    }

    private func processVideoImage(_ image: CIImage) throws -> CIImage {
        let size = try config.videoTargetSize(for: image.extent.size)
        var image = image.toSRGB().resampled(to: size, method: .bicubic)
        if config.videoDoNormalize {
            image = image.normalized(
                mean: config.videoImageMeanTuple, std: config.videoImageStdTuple)
        }
        return image
    }

    private func processVideoFrame(_ image: CIImage) throws -> MLXArray {
        let pixels = MediaProcessing.asMLXArray(try processVideoImage(image))
        eval(pixels)
        return pixels
    }

    private func prepareVideos(_ videos: [UserInput.Video], processing: UserInput.Processing?)
        async throws -> (
            pixels: MLXArray, frameCounts: [Int], timestamps: [[Double]], frames: [THW]
        )
    {
        var pixels: [MLXArray] = []
        var counts: [Int] = []
        var timestamps: [[Double]] = []
        var geometry: [THW] = []
        for video in videos {
            if case .frames(let frames) = video.source {
                guard !frames.isEmpty else { throw Gemma4Error.emptyVideoFrames }
                for frame in frames {
                    let image = try frame.image.asCIImage()
                    let framePixels = try autoreleasepool { try processVideoFrame(image) }
                    pixels.append(framePixels)
                    geometry.append(THW(1, framePixels.dim(2), framePixels.dim(3)))
                }
                counts.append(frames.count)
                timestamps.append(frames.map { $0.timeStamp.seconds })
            } else {
                let sequence = try await MediaProcessing.asProcessedSequence(
                    video, processing: processing?.video ?? .init(),
                    targetFPS: { _ in Double(config.videoFps) }, maxFrames: config.videoMaxFrames
                ) { frame in
                    let image = try processVideoImage(frame.image.asCIImage())
                    return VideoFrame(image: .ciImage(image), timeStamp: frame.timeStamp)
                }
                for frame in sequence.frames {
                    eval(frame)
                    pixels.append(frame)
                    geometry.append(THW(1, frame.dim(2), frame.dim(3)))
                }
                counts.append(sequence.frames.count)
                timestamps.append(sequence.timestamps.map { $0.seconds })
            }
        }
        guard !pixels.isEmpty else { throw Gemma4Error.emptyVideoFrames }
        let maxHeight = geometry.map(\.h).max()!
        let maxWidth = geometry.map(\.w).max()!
        let paddedPixels = zip(pixels, geometry).map { pixels, frame in
            MLX.padded(
                pixels,
                widths: [0, 0, .init((0, maxHeight - frame.h)), .init((0, maxWidth - frame.w))])
        }
        return (concatenated(paddedPixels), counts, timestamps, geometry)
    }

    public func processVideos(_ videos: [UserInput.Video], processing: UserInput.Processing?)
        async throws -> (pixels: MLXArray, frameCounts: [Int])
    {
        let prepared = try await prepareVideos(videos, processing: processing)
        return (prepared.pixels, prepared.frameCounts)
    }

    private func videoToken() throws -> (id: Int, text: String) {
        for text in [Self.videoPlaceholder, "<video>"] {
            if let id = tokenizer.convertTokenToId(text), tokenizer.convertIdToToken(id) == text {
                if let configured = config.videoTokenId, configured != id {
                    if let actual = tokenizer.convertIdToToken(configured),
                        [Self.videoPlaceholder, "<video>"].contains(actual)
                    {
                        return (configured, actual)
                    }
                    throw Gemma4Error.videoTokenIdMismatch(configuration: configured, tokenizer: id)
                }
                return (id, text)
            }
        }
        throw Gemma4Error.missingVideoTokenMapping
    }

    private func validateVideoBoundaryToken(id: Int, configurationKey: String) throws {
        guard let token = tokenizer.convertIdToToken(id),
            tokenizer.convertTokenToId(token) == id
        else {
            throw Gemma4Error.missingVideoBoundaryTokenMapping(
                configurationKey: configurationKey, tokenId: id)
        }
    }

    private func timestampTokens(_ seconds: Double) throws -> [Int] {
        guard seconds.isFinite, seconds >= 0, seconds < Double(Int64.max) / 1_000_000 else {
            throw VLMError.processing("Video frames must use finite, nonnegative timestamps.")
        }
        let total = Int64((seconds * 1_000_000).rounded())
        let timestamp = String(
            format: "%02lld:%02lld.%06lld", locale: Locale(identifier: "en_US_POSIX"),
            total / 60_000_000, (total / 1_000_000) % 60, total % 1_000_000)
        return tokenizer.encode(text: timestamp + " ", addSpecialTokens: false)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let videoMapping = input.videos.isEmpty ? nil : try videoToken()
        let messages = Gemma4MessageGenerator(
            videoPlaceholder: videoMapping?.text ?? Self.videoPlaceholder
        ).generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        print("[Omni Debug] Full decoded prompt: \(tokenizer.decode(tokenIds: promptTokens))")

        if input.images.isEmpty, input.videos.isEmpty, input.audio.isEmpty {
            // No mask: one unpadded sequence needs none, and a mask stops prompt-cache reuse.
            return LMInput(tokens: MLXArray(promptTokens).expandedDimensions(axis: 0))
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(image: $0.asCIImage(), processing: input.processing)
            }
            let frames = imagePixelsAndFrames.map { $0.1 }

            // Each image keeps its own aspect-preserving size. ProcessedImage
            // carries one array, so zero-pad every image onto the largest
            // canvas in the request; the model slices the real regions back
            // out using frames.
            let maxHeight = frames.map(\.h).max() ?? 0
            let maxWidth = frames.map(\.w).max() ?? 0
            let paddedPixels = imagePixelsAndFrames.map { pixels, frame in
                frame.h == maxHeight && frame.w == maxWidth
                    ? pixels
                    : MLX.padded(
                        pixels,
                        widths: [
                            0, 0, .init((0, maxHeight - frame.h)), .init((0, maxWidth - frame.w)),
                        ])
            }
            processedImage = LMInput.ProcessedImage(
                pixels: concatenated(paddedPixels), frames: frames)

            // Expand the i-th image placeholder to that image's soft token
            // count: numPatches / poolingKernelSize².
            let softTokenCounts = frames.map { config.softTokenCount(height: $0.h, width: $0.w) }
            var expandedTokens: [Int] = []
            var imageIndex = 0
            for token in promptTokens {
                if token == config.imageTokenId {
                    guard imageIndex < softTokenCounts.count else {
                        throw Gemma4Error.imagePlaceholderMismatch(
                            images: softTokenCounts.count, placeholders: imageIndex + 1)
                    }
                    expandedTokens.append(config.boiTokenId)
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: config.imageTokenId,
                            count: softTokenCounts[imageIndex]))
                    if let eoiTokenId = config.eoiTokenId {
                        expandedTokens.append(eoiTokenId)
                    }
                    imageIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }

        var processedVideo: LMInput.ProcessedVideo?
        if let videoMapping {
            try validateVideoBoundaryToken(id: config.boiTokenId, configurationKey: "boi_token_id")
            if let end = config.eoiTokenId {
                try validateVideoBoundaryToken(id: end, configurationKey: "eoi_token_id")
            }
            let prepared = try await prepareVideos(input.videos, processing: input.processing)
            processedVideo = .init(pixels: prepared.pixels, frames: prepared.frames)
            let placeholders = promptTokens.filter { $0 == videoMapping.id }.count
            guard placeholders == input.videos.count else {
                throw Gemma4Error.videoPlaceholderCountMismatch(
                    videos: input.videos.count, placeholders: placeholders)
            }
            var expanded: [Int] = []
            var videoIndex = 0
            var frameIndex = 0
            for token in promptTokens {
                if token == videoMapping.id {
                    for timestamp in prepared.timestamps[videoIndex] {
                        expanded.append(contentsOf: try timestampTokens(timestamp))
                        expanded.append(config.boiTokenId)
                        expanded.append(
                            contentsOf: Array(
                                repeating: videoMapping.id,
                                count: config.videoSoftTokenCount(
                                    height: prepared.frames[frameIndex].h,
                                    width: prepared.frames[frameIndex].w)))
                        if let end = config.eoiTokenId { expanded.append(end) }
                        expanded.append(
                            contentsOf: tokenizer.encode(text: " ", addSpecialTokens: false))
                        frameIndex += 1
                    }
                    videoIndex += 1
                } else {
                    expanded.append(token)
                }
            }
            promptTokens = expanded
        }

        var processedAudio: LMInput.ProcessedAudio? = nil
        if let audioInput = input.audio.first {
            let samples = try MediaProcessing.extractAudioSamples(from: audioInput)
            print(
                "[Omni Debug] Extracted \(samples.count) float samples. Min: \(samples.min() ?? 0.0), Max: \(samples.max() ?? 0.0)"
            )
            let processor = Gemma4AudioFeatureExtractor(
                featureSize: 128,
                samplingRate: 16000,
                frameLengthMs: 20.0,
                hopLengthMs: 10.0,
                melFloor: 1e-3
            )
            let (features, mask) = processor.extract(waveform: samples)
            print(
                "[Omni Debug] Extracted \(features.shape) features. Mean: \(features.mean().item(Float.self)), Min: \(features.min().item(Float.self)), Max: \(features.max().item(Float.self))"
            )

            // Expected audio token count: mirrors the audio tower's two stride-2 conv2d layers,
            // each with 1-sample symmetric padding (MLX.padded [1,1] in time axis).
            // Layer formula: ceil(T/2) applied twice → ceil(ceil(numMelFrames/2)/2).
            // This avoids off-by-one vs. duration/40ms approximation.
            let numMelFrames = features.dim(1)
            let afterLayer0 = (numMelFrames + 1) / 2  // ceil(T/2)
            var expectedAudioTokens = (afterLayer0 + 1) / 2  // ceil(L1/2)
            expectedAudioTokens = min(expectedAudioTokens, 750)  // audio_seq_length cap

            let seqLength = features.dim(1)
            processedAudio = LMInput.ProcessedAudio(
                features: features, mask: mask, seqLengths: [seqLength])

            let audioTokenId = config.audioTokenId ?? 258881
            let gemmaBoa = 256000  // <|audio|>
            let gemmaEoa = 258883  // <audio|>

            var audioPadding = [gemmaBoa]
            audioPadding.append(
                contentsOf: Array(repeating: audioTokenId, count: expectedAudioTokens))
            audioPadding.append(gemmaEoa)

            let targetIdx =
                promptTokens.firstIndex(of: gemmaBoa) ?? promptTokens.firstIndex(of: audioTokenId)
            print(
                "[Omni Debug] Target Index found at: \(String(describing: targetIdx)), gemmaBoa: \(gemmaBoa), audioTokenId: \(audioTokenId)"
            )

            var expandedTokens = promptTokens
            expandedTokens.removeAll(where: {
                $0 == gemmaBoa || $0 == gemmaEoa || $0 == audioTokenId
            })

            if let insertIdx = targetIdx {
                print(
                    "[Omni Debug] Inserting audioPadding (\(audioPadding.count) tokens) at \(insertIdx)"
                )
                expandedTokens.insert(contentsOf: audioPadding, at: insertIdx)
            } else {
                print("[Omni Debug] Inserting audioPadding (\(audioPadding.count) tokens) at 0")
                expandedTokens.insert(contentsOf: audioPadding, at: 0)
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask), image: processedImage,
            video: processedVideo, audio: processedAudio)
    }
}

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let maxSoftTokens: Int
    public let patchSize: Int
    public let poolingKernelSize: Int

    public let imageTokenId: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let boaTokenId: Int
    public let eoaTokenId: Int
    public let audioTokenId: Int?

    public let videoTokenId: Int?
    public let videoSoftTokensPerFrame: Int
    public let videoMaxFrames: Int
    public let videoFps: Float
    public let videoFrameSize: Gemma3ProcessorConfiguration.ImageSize?
    private let videoProcessor: ImageProcessorConfiguration?
    public let boiTokenString: String
    public let eoiTokenString: String
    public var videoSeqLength: Int { videoSoftTokensPerFrame }
    public var videoFrameFixedSize: CGSize { videoFixedSize }

    /// Image keys nested under `image_processor` in processor_config.json.
    /// Repos that ship a flat preprocessor_config.json put the same keys at
    /// the top level, which wins when both are present.
    private struct ImageProcessorConfiguration: Codable, Sendable {
        let doNormalize: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let imageSeqLength: Int?
        let maxSoftTokens: Int?
        let patchSize: Int?
        let poolingKernelSize: Int?
        let numFrames: Int?

        enum CodingKeys: String, CodingKey {
            case doNormalize = "do_normalize"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case imageSeqLength = "image_seq_length"
            case maxSoftTokens = "max_soft_tokens"
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
            case numFrames = "num_frames"
        }
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case maxSoftTokens = "max_soft_tokens"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case imageProcessor = "image_processor"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case videoTokenId = "video_token_id"
        case videoSoftTokensPerFrame = "video_soft_tokens_per_frame"
        case videoMaxFrames = "video_max_frames"
        case videoFps = "video_fps"
        case videoFrameSize = "video_frame_size"
        case videoProcessor = "video_processor"
        case videoSeqLength = "video_seq_length"
        case boiTokenString = "boi_token"
        case eoiTokenString = "eoi_token"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
        case audioTokenId = "audio_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(
            ImageProcessorConfiguration.self, forKey: CodingKeys.imageProcessor)
        videoProcessor = try c.decodeIfPresent(
            ImageProcessorConfiguration.self, forKey: CodingKeys.videoProcessor)
        processorClass = try c.decode(String.self, forKey: CodingKeys.processorClass)
        doNormalize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doNormalize)
            ?? nested?.doNormalize ?? false
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageMean)
            ?? nested?.imageMean ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageStd)
            ?? nested?.imageStd ?? [0.5, 0.5, 0.5]
        imageSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageSeqLength)
            ?? nested?.imageSeqLength ?? 280
        maxSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxSoftTokens)
            ?? nested?.maxSoftTokens ?? 280
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize)
            ?? nested?.patchSize ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize)
            ?? nested?.poolingKernelSize ?? 3
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId) ?? 258_884
        videoSoftTokensPerFrame =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoSoftTokensPerFrame)
            ?? c.decodeIfPresent(Int.self, forKey: CodingKeys.videoSeqLength)
            ?? videoProcessor?.maxSoftTokens ?? 70
        videoMaxFrames =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoMaxFrames)
            ?? videoProcessor?.numFrames ?? 32
        videoFps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.videoFps) ?? 1
        videoFrameSize = try c.decodeIfPresent(
            Gemma3ProcessorConfiguration.ImageSize.self, forKey: CodingKeys.videoFrameSize)
        boiTokenString =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.boiTokenString)
            ?? "<start_of_image>"
        eoiTokenString =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.eoiTokenString)
            ?? "<end_of_image>"
        boaTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boaTokenId) ?? 256_000
        eoaTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoaTokenId) ?? 258_883
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId) ?? 258_881
    }

    public func encode(to encoder: any Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(processorClass, forKey: CodingKeys.processorClass)
        try c.encode(doNormalize, forKey: CodingKeys.doNormalize)
        try c.encode(imageMean, forKey: CodingKeys.imageMean)
        try c.encode(imageStd, forKey: CodingKeys.imageStd)
        try c.encode(imageSeqLength, forKey: CodingKeys.imageSeqLength)
        try c.encode(maxSoftTokens, forKey: CodingKeys.maxSoftTokens)
        try c.encode(patchSize, forKey: CodingKeys.patchSize)
        try c.encode(poolingKernelSize, forKey: CodingKeys.poolingKernelSize)
        try c.encode(imageTokenId, forKey: CodingKeys.imageTokenId)
        try c.encode(boiTokenId, forKey: CodingKeys.boiTokenId)
        try c.encodeIfPresent(eoiTokenId, forKey: CodingKeys.eoiTokenId)
        try c.encode(boaTokenId, forKey: CodingKeys.boaTokenId)
        try c.encode(eoaTokenId, forKey: CodingKeys.eoaTokenId)
        try c.encodeIfPresent(audioTokenId, forKey: CodingKeys.audioTokenId)
        try c.encodeIfPresent(videoTokenId, forKey: CodingKeys.videoTokenId)
        try c.encode(videoSoftTokensPerFrame, forKey: CodingKeys.videoSoftTokensPerFrame)
        try c.encode(videoMaxFrames, forKey: CodingKeys.videoMaxFrames)
        try c.encode(videoFps, forKey: CodingKeys.videoFps)
        try c.encodeIfPresent(videoFrameSize, forKey: CodingKeys.videoFrameSize)
        try c.encodeIfPresent(videoProcessor, forKey: CodingKeys.videoProcessor)
        try c.encode(boiTokenString, forKey: CodingKeys.boiTokenString)
        try c.encode(eoiTokenString, forKey: CodingKeys.eoiTokenString)
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var videoPatchSize: Int { videoProcessor?.patchSize ?? patchSize }
    public var videoPoolingKernelSize: Int {
        videoProcessor?.poolingKernelSize ?? poolingKernelSize
    }
    public var videoDoNormalize: Bool { videoProcessor?.doNormalize ?? doNormalize }

    public var videoImageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        let mean = videoProcessor?.imageMean ?? imageMean
        return (mean[0], mean[1], mean[2])
    }

    public var videoImageStdTuple: (CGFloat, CGFloat, CGFloat) {
        let standardDeviation = videoProcessor?.imageStd ?? imageStd
        return (standardDeviation[0], standardDeviation[1], standardDeviation[2])
    }

    public func videoSoftTokenCount(height: Int, width: Int) -> Int {
        let poolingSide = videoPatchSize * videoPoolingKernelSize
        return (height / poolingSide) * (width / poolingSide)
    }

    public func videoTargetSize(for imageSize: CGSize) throws -> CGSize {
        guard imageSize.width.isFinite, imageSize.height.isFinite,
            imageSize.width > 0, imageSize.height > 0,
            videoSoftTokensPerFrame > 0, videoPatchSize > 0, videoPoolingKernelSize > 0
        else {
            throw VLMError.processing(
                "Gemma4 video requires positive dimensions and geometry settings.")
        }
        let size =
            videoFrameSize.map { CGSize(width: $0.width, height: $0.height) }
            ?? Self.aspectPreservingTargetSize(
                for: imageSize, budget: videoSoftTokensPerFrame,
                patchSize: videoPatchSize, poolingKernelSize: videoPoolingKernelSize)
        let poolingSide = videoPatchSize * videoPoolingKernelSize
        guard size.width > 0, size.height > 0,
            Int(size.width) % poolingSide == 0, Int(size.height) % poolingSide == 0,
            videoSoftTokenCount(height: Int(size.height), width: Int(size.width))
                <= videoSoftTokensPerFrame
        else {
            throw VLMError.processing(
                "Gemma4 video frame size must divide into complete pooling cells within the configured token maximum."
            )
        }
        return size
    }

    /// Soft tokens the vision tower produces for an image of the given
    /// (already resized) pixel dimensions.
    public func softTokenCount(height: Int, width: Int) -> Int {
        ((height / patchSize) * (width / patchSize)) / (poolingKernelSize * poolingKernelSize)
    }

    /// Port of the Python Gemma4ImageProcessor's aspect-ratio preserving
    /// resize: the largest dimensions that (a) stay within the patch budget
    /// maxSoftTokens * poolingKernelSize², and (b) keep both sides divisible
    /// by poolingKernelSize * patchSize, so the pooling kernel is exact and
    /// the pooled grid covers the image fully at any aspect ratio.
    ///
    /// Note the config's `size` entry is deliberately ignored, as in the
    /// Python reference — models ship a vestigial 224x224 there.
    ///
    /// - Parameters:
    ///   - imageSize: Original image dimensions.
    ///   - budget: Soft tokens to fit; defaults to `maxSoftTokens`.
    public func aspectPreservingTargetSize(for imageSize: CGSize, budget: Int? = nil) -> CGSize {
        Self.aspectPreservingTargetSize(
            for: imageSize, budget: budget ?? maxSoftTokens,
            patchSize: patchSize, poolingKernelSize: poolingKernelSize)
    }

    private static func aspectPreservingTargetSize(
        for imageSize: CGSize, budget maxSoftTokens: Int, patchSize: Int, poolingKernelSize: Int
    ) -> CGSize {
        let kernelArea = poolingKernelSize * poolingKernelSize
        let maxPatches = maxSoftTokens * kernelArea
        let sideMultiple = poolingKernelSize * patchSize
        let height = Double(imageSize.height)
        let width = Double(imageSize.width)

        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / max(height * width, 1)).squareRoot()
        var targetHeight = Int((factor * height / Double(sideMultiple)).rounded(.down))
        var targetWidth = Int((factor * width / Double(sideMultiple)).rounded(.down))

        // One side can floor to zero for extreme aspect ratios (both cannot:
        // their product is pinned near maxPatches, far above 1). Clamp it to
        // one pooling cell and cap the long side at the full token budget.
        let maxSideLength = maxSoftTokens
        if targetHeight == 0 {
            targetHeight = 1
            targetWidth = min(Int((width / max(height, 1)).rounded(.down)), maxSideLength)
            targetWidth = max(targetWidth, 1)
        } else if targetWidth == 0 {
            targetWidth = 1
            targetHeight = min(Int((height / max(width, 1)).rounded(.down)), maxSideLength)
            targetHeight = max(targetHeight, 1)
        }

        return CGSize(
            width: targetWidth * sideMultiple, height: targetHeight * sideMultiple)
    }

    public var videoFixedSize: CGSize {
        if let videoFrameSize {
            return CGSize(width: videoFrameSize.width, height: videoFrameSize.height)
        }
        return Self.aspectPreservingTargetSize(
            for: CGSize(width: 1, height: 1), budget: videoSoftTokensPerFrame,
            patchSize: videoPatchSize, poolingKernelSize: videoPoolingKernelSize)
    }

}

public struct Gemma4UnifiedProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let doResize: Bool
    public let doRescale: Bool
    public let rescaleFactor: CGFloat
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let audioSeqLength: Int
    public let audioMsPerToken: Int
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let modelPatchSize: Int
    public let maxSoftTokens: Int
    public let size: Gemma3ProcessorConfiguration.ImageSize?
    public let imageTokenId: Int
    public let audioTokenId: Int
    public let videoTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?

    private struct ImageProcessorConfiguration: Decodable, Sendable {
        let doResize: Bool?
        let doRescale: Bool?
        let rescaleFactor: CGFloat?
        let doNormalize: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let patchSize: Int?
        let poolingKernelSize: Int?
        let modelPatchSize: Int?
        let maxSoftTokens: Int?
        let numSoftTokens: Int?
        let size: Gemma3ProcessorConfiguration.ImageSize?

        enum CodingKeys: String, CodingKey {
            case doResize = "do_resize"
            case doRescale = "do_rescale"
            case rescaleFactor = "rescale_factor"
            case doNormalize = "do_normalize"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
            case modelPatchSize = "model_patch_size"
            case maxSoftTokens = "max_soft_tokens"
            case numSoftTokens = "num_soft_tokens"
            case size
        }
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
        case doResize = "do_resize"
        case doRescale = "do_rescale"
        case rescaleFactor = "rescale_factor"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case audioSeqLength = "audio_seq_length"
        case audioMsPerToken = "audio_ms_per_token"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case modelPatchSize = "model_patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case numSoftTokens = "num_soft_tokens"
        case size
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let imageProcessor = try c.decodeIfPresent(
            ImageProcessorConfiguration.self, forKey: CodingKeys.imageProcessor)

        processorClass =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.processorClass)
            ?? "Gemma4UnifiedProcessor"
        doResize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doResize)
            ?? imageProcessor?.doResize
            ?? true
        doRescale =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doRescale)
            ?? imageProcessor?.doRescale
            ?? true
        rescaleFactor =
            try c.decodeIfPresent(CGFloat.self, forKey: CodingKeys.rescaleFactor)
            ?? imageProcessor?.rescaleFactor
            ?? (1.0 / 255.0)
        doNormalize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doNormalize)
            ?? imageProcessor?.doNormalize
            ?? false
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageMean)
            ?? imageProcessor?.imageMean
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageStd)
            ?? imageProcessor?.imageStd
            ?? [0.5, 0.5, 0.5]
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize)
            ?? imageProcessor?.patchSize
            ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize)
            ?? imageProcessor?.poolingKernelSize
            ?? 3
        modelPatchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.modelPatchSize)
            ?? imageProcessor?.modelPatchSize
            ?? patchSize * poolingKernelSize
        maxSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxSoftTokens)
            ?? c.decodeIfPresent(Int.self, forKey: CodingKeys.numSoftTokens)
            ?? imageProcessor?.maxSoftTokens
            ?? imageProcessor?.numSoftTokens
            ?? 280
        imageSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageSeqLength) ?? maxSoftTokens
        audioSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSeqLength) ?? 750
        audioMsPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioMsPerToken) ?? 40
        size =
            try c.decodeIfPresent(
                Gemma3ProcessorConfiguration.ImageSize.self, forKey: CodingKeys.size)
            ?? imageProcessor?.size
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId) ?? 258_881
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId) ?? 258_884
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var fixedSize: CGSize {
        let patchesPerSide = max(1, Int(floor(sqrt(Double(maxSoftTokens)))))
        let side = patchesPerSide * patchSize * poolingKernelSize
        return CGSize(width: side, height: side)
    }

    public func aspectRatioPreservingSize(for imageSize: CGSize) throws -> CGSize {
        let width = max(1, Int(ceil(imageSize.width)))
        let height = max(1, Int(ceil(imageSize.height)))
        let sideMultiple = max(1, patchSize * poolingKernelSize)
        let maxTokens = max(1, maxSoftTokens)

        let targetPixels = Double(maxTokens * sideMultiple * sideMultiple)
        let resizeFactor = sqrt(targetPixels / Double(width * height))

        var targetWidth =
            Int(floor(Double(width) * resizeFactor / Double(sideMultiple))) * sideMultiple
        var targetHeight =
            Int(floor(Double(height) * resizeFactor / Double(sideMultiple))) * sideMultiple

        if targetWidth == 0 && targetHeight == 0 {
            throw VLMError.processing("Image is too small to resize for Gemma4 unified vision.")
        } else if targetHeight == 0 {
            targetHeight = sideMultiple
            targetWidth = max(
                sideMultiple,
                min(
                    maxTokens * sideMultiple,
                    Int(floor(Double(width) / Double(height))) * sideMultiple))
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            targetHeight = max(
                sideMultiple,
                min(
                    maxTokens * sideMultiple,
                    Int(floor(Double(height) / Double(width))) * sideMultiple))
        }

        return CGSize(width: targetWidth, height: targetHeight)
    }
}

public struct Gemma4UnifiedProcessor: UserInputProcessor {
    private let config: Gemma4UnifiedProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4UnifiedProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func patchify(_ pixelValues: MLXArray) -> (MLXArray, MLXArray, Int, THW) {
        let channels = pixelValues.dim(1)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)
        let patchHeight = height / config.modelPatchSize
        let patchWidth = width / config.modelPatchSize
        let realCount = min(patchHeight * patchWidth, config.maxSoftTokens)
        let patchDim = config.modelPatchSize * config.modelPatchSize * channels

        var patches = pixelValues.reshaped(
            1, channels, patchHeight, config.modelPatchSize, patchWidth, config.modelPatchSize)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(patchHeight * patchWidth, patchDim)
        if realCount < patches.dim(0) {
            patches = patches[..<realCount, 0...]
        }
        if realCount < config.maxSoftTokens {
            patches = padded(patches, widths: [.init((0, config.maxSoftTokens - realCount)), 0])
        }

        var positionValues: [Int32] = []
        positionValues.reserveCapacity(config.maxSoftTokens * 2)
        var emitted = 0
        for y in 0 ..< patchHeight {
            for x in 0 ..< patchWidth where emitted < realCount {
                positionValues.append(Int32(x))
                positionValues.append(Int32(y))
                emitted += 1
            }
        }
        while emitted < config.maxSoftTokens {
            positionValues.append(-1)
            positionValues.append(-1)
            emitted += 1
        }
        let positions = MLXArray(positionValues, [config.maxSoftTokens, 2])
        return (patches, positions, realCount, THW(1, height, width))
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        pixels: MLXArray, positionIds: MLXArray, tokenCounts: [Int], frames: [THW]
    ) {
        var patchRows: [MLXArray] = []
        var positionRows: [MLXArray] = []
        var tokenCounts: [Int] = []
        var frames: [THW] = []

        for image in images {
            let processedImage = MediaProcessing.apply(image, processing: processing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage =
                if config.doResize {
                    try MediaProcessing.resampleBicubic(
                        srgbImage,
                        to: config.aspectRatioPreservingSize(for: srgbImage.extent.size))
                } else {
                    srgbImage
                }

            var pixelValues = MediaProcessing.asMLXArray(resizedImage)
            let rescaleMultiplier = Float(config.doRescale ? config.rescaleFactor * 255 : 255)
            if rescaleMultiplier != 1 {
                pixelValues = pixelValues * MLXArray(rescaleMultiplier, dtype: pixelValues.dtype)
            }
            if config.doNormalize {
                let mean = MLXArray(
                    config.imageMean.map { Float($0) }, [1, config.imageMean.count, 1, 1]
                )
                .asType(pixelValues.dtype)
                let std = MLXArray(
                    config.imageStd.map { Float($0) }, [1, config.imageStd.count, 1, 1]
                )
                .asType(pixelValues.dtype)
                pixelValues = (pixelValues - mean) / std
            }
            let (patches, positions, tokenCount, frame) = patchify(pixelValues)
            patchRows.append(patches)
            positionRows.append(positions)
            tokenCounts.append(tokenCount)
            frames.append(frame)
        }

        return (
            pixels: stacked(patchRows, axis: 0),
            positionIds: stacked(positionRows, axis: 0),
            tokenCounts: tokenCounts,
            frames: frames
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        if input.images.isEmpty {
            // No mask: one unpadded sequence needs none, and a mask stops prompt-cache reuse.
            return LMInput(tokens: MLXArray(promptTokens).expandedDimensions(axis: 0))
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageData = try preprocess(
                images: input.images.map { try $0.asCIImage() },
                processing: input.processing
            )
            processedImage = LMInput.ProcessedImage(
                pixels: imageData.pixels,
                positionIds: imageData.positionIds,
                frames: imageData.frames
            )

            var imageIndex = 0
            var expandedTokens: [Int] = []
            expandedTokens.reserveCapacity(
                promptTokens.count + imageData.tokenCounts.reduce(0, +))
            for token in promptTokens {
                if token == config.imageTokenId {
                    let count =
                        imageIndex < imageData.tokenCounts.count
                        ? imageData.tokenCounts[imageIndex]
                        : config.imageSeqLength
                    expandedTokens.append(config.boiTokenId)
                    expandedTokens.append(
                        contentsOf: Array(repeating: config.imageTokenId, count: count))
                    if let eoiTokenId = config.eoiTokenId {
                        expandedTokens.append(eoiTokenId)
                    }
                    imageIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(text: .init(tokens: promptArray, mask: mask), image: processedImage)
    }
}

// MARK: - Chat conventions

extension Gemma4 {
    public var toolCallFormat: ToolCallFormat? { .gemma4 }
}

extension Gemma4Unified {
    public var toolCallFormat: ToolCallFormat? { .gemma4 }
}

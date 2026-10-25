import MLX
import MLXLMCommon
import Testing

extension MLXTestingSuite {
    @Suite
    struct ForkCacheCompatibilityTests {
        @Test func compressedHistoryParticipatesInAttention() {
            let previousMemoryLimit = Memory.memoryLimit
            let previousCacheLimit = Memory.cacheLimit
            Memory.memoryLimit = 64 * 1024 * 1024
            Memory.cacheLimit = 0
            defer {
                Memory.clearCache()
                Memory.cacheLimit = previousCacheLimit
                Memory.memoryLimit = previousMemoryLimit
            }

            let cache = KVCacheSimple()
            cache.step = 1
            cache.turboQuantEnabled = true
            cache.turboMinActivationTokens = 1
            cache.turboHotWindowSize = 1
            let keys = MLXArray(0 ..< 384).asType(.float32).reshaped(1, 1, 3, 128) / 384
            let values = keys + 1
            _ = cache.update(keys: keys, values: values)
            #expect(cache.compressedOffset == 2)
            #expect(cache.offset == 1)
            let retainedState = cache.state
            eval(retainedState)

            cache.turboMinActivationTokens = Int.max
            let nextKeys = MLXArray.zeros([1, 1, 1, 128])
            let nextValues = MLXArray.zeros([1, 1, 1, 128])
            let queries = MLXArray.ones([1, 1, 1, 128])
            let expected = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: concatenated([retainedState[0], nextKeys], axis: 2),
                values: concatenated([retainedState[1], nextValues], axis: 2),
                scale: 0.125, mask: .none)
            let output = attentionWithCacheUpdate(
                queries: queries, keys: nextKeys, values: nextValues,
                cache: cache, scale: 0.125)
            eval(output, expected)

            #expect(allClose(output, expected, atol: 1e-5).item(Bool.self))
            #expect(output.mean().item(Float.self) > 0.5)
            #expect(cache.compressedOffset == 2)
        }
    }
}

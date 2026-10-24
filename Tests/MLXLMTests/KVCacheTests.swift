import Foundation
import MLX
import MLXLMCommon
import Testing

extension MLXTestingSuite {
    @Suite
    struct KVCacheTests {
    private static let cacheCreators: [@Sendable () -> any KVCache] = [
        { KVCacheSimple() },
        { RotatingKVCache(maxSize: 32) },
        { QuantizedKVCache() },
        { ChunkedKVCache(chunkSize: 16) },
        { ArraysCache(size: 2) },
        { MambaCache() },
    ]

    @Test(
        .serialized,
        arguments: cacheCreators)
    func testCacheSerialization(creator: (() -> any KVCache)) async throws {
        let cache = (0 ..< 10).map { _ in creator() }
        let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
        let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
        for item in cache {
            switch item {
            case let arrays as ArraysCache:
                arrays[0] = keys
                arrays[1] = values
            case let quantized as QuantizedKVCache:
                _ = quantized.updateQuantized(keys: keys, values: values)
            default:
                _ = item.update(keys: keys, values: values)
            }
        }

        let url = tempURL()

        try savePromptCache(url: url, cache: cache, metadata: [:])
        let (loadedCache, _) = try loadPromptCache(url: url)

        #expect(cache.count == loadedCache.count)
        for (lhs, rhs) in zip(cache, loadedCache) {
            #expect(type(of: lhs) == type(of: rhs))
            #expect(lhs.metaState == rhs.metaState)
            assertArraysClose(lhs.state, rhs.state)
        }
    }

    @Test func testPromptCacheStateRoundTrip() throws {
        let cache = KVCacheSimple()
        let keys = MLXArray.ones([1, 2, 4, 8], dtype: .bfloat16)
        let values = MLXArray.zeros([1, 2, 4, 8], dtype: .bfloat16)
        _ = cache.update(keys: keys, values: values)

        let ropeDeltasKey = LMOutput.Key<MLXArray>("test.ropeDeltas")
        let positionsKey = LMOutput.Key<MLXArray>("test.positionIds")
        let ropeDeltas = MLXArray([Int32(3), 5, 7]).reshaped([1, 3])
        let positions = MLXArray([Int32(11), 13, 17, 19]).reshaped([2, 2])
        var state = LMOutput.State()
        state[ropeDeltasKey] = ropeDeltas
        state[positionsKey] = positions

        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(
            url: url, cache: [cache], metadata: ["source": "test"], state: state)

        let snapshot = try loadPromptCacheSnapshot(url: url)
        let restoredRopeDeltas = try #require(snapshot.state?[ropeDeltasKey])
        let restoredPositions = try #require(snapshot.state?[positionsKey])

        #expect(snapshot.metadata == ["source": "test"])
        #expect(restoredRopeDeltas.dtype == ropeDeltas.dtype)
        #expect(restoredRopeDeltas.shape == ropeDeltas.shape)
        #expect(allClose(restoredRopeDeltas, ropeDeltas, rtol: 0, atol: 0).item(Bool.self))
        #expect(restoredPositions.dtype == positions.dtype)
        #expect(restoredPositions.shape == positions.shape)
        #expect(allClose(restoredPositions, positions, rtol: 0, atol: 0).item(Bool.self))

        let (storedArrays, storedMetadata) = try loadArraysAndMetadata(url: url)
        #expect(storedArrays["__mlx_lm_state_tensor_0"] != nil)
        #expect(storedArrays["__mlx_lm_state_tensor_1"] != nil)
        #expect(
            storedArrays.keys.allSatisfy {
                Int($0.split(separator: ".")[0]) != nil
                    || $0.hasPrefix("__mlx_lm_state_tensor_")
            })
        #expect(storedMetadata["1.__mlx_lm_state_0_key"] == "test.positionIds")
        #expect(storedMetadata["1.__mlx_lm_state_1_key"] == "test.ropeDeltas")
        #expect(storedMetadata["2.0"] == "__mlx_lm_state_v1__:KVCache")
        #expect(throws: KVCacheError.self) {
            try loadPromptCache(url: url)
        }
    }

    @Test func testLegacyPromptCacheLoadsWithNilState() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache])
        let snapshot = try loadPromptCacheSnapshot(url: url)
        let (legacyCache, legacyMetadata) = try loadPromptCache(url: url)
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)

        #expect(snapshot.cache.count == 1)
        #expect(snapshot.state == nil)
        #expect(legacyCache.count == 1)
        #expect(legacyMetadata.isEmpty)
        #expect(arrays.keys.allSatisfy { Int($0.split(separator: ".")[0]) != nil })
        #expect(!metadata.keys.contains { $0.contains("__mlx_lm_state_") })
        #expect(metadata["2.0"] == "KVCache")
    }

    @Test func testLoadedPromptCacheNoLongerReadsItsFile() throws {
        let cache = KVCacheSimple()
        let keys = MLXArray(0 ..< 32).reshaped([1, 2, 4, 4]).asType(.float32)
        let values = -keys
        _ = cache.update(keys: keys, values: values)
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [cache])

        // Saving truncates the file before it evaluates what it writes.
        let snapshot = try loadPromptCacheSnapshot(url: url)
        try savePromptCache(url: url, cache: snapshot.cache)

        let restored = try loadPromptCacheSnapshot(url: url)
        assertArraysClose(restored.cache[0].state, [keys, values])
    }

    @Test func testEmptyPromptCacheStateUsesTheLegacyFormat() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let legacyURL = tempURL()
        let emptyStateURL = tempURL()
        defer {
            try? FileManager.default.removeItem(at: legacyURL)
            try? FileManager.default.removeItem(at: emptyStateURL)
        }

        try savePromptCache(url: legacyURL, cache: [cache])
        try savePromptCache(url: emptyStateURL, cache: [cache], state: LMOutput.State())
        let snapshot = try loadPromptCacheSnapshot(url: emptyStateURL)
        let (legacyCache, _) = try loadPromptCache(url: emptyStateURL)
        let (legacyArrays, legacyMetadata) = try loadArraysAndMetadata(url: legacyURL)
        let (emptyStateArrays, emptyStateMetadata) = try loadArraysAndMetadata(url: emptyStateURL)

        #expect(snapshot.state == nil)
        #expect(legacyCache.count == 1)
        #expect(Set(emptyStateArrays.keys) == Set(legacyArrays.keys))
        #expect(emptyStateMetadata == legacyMetadata)
    }

    @Test func testPromptCacheStateRejectsReservedUserMetadata() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: KVCacheError.self) {
            try savePromptCache(
                url: url, cache: [cache],
                metadata: ["__mlx_lm_state_version": "caller-owned"])
        }
    }

    @Test func testPromptCacheStateRejectsUnsupportedValues() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let unsupportedKey = LMOutput.Key<Bool>("test.unsupported")
        var state = LMOutput.State()
        state[unsupportedKey] = true
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: LMOutput.State.SerializationError.self) {
            try savePromptCache(url: url, cache: [cache], state: state)
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func testPromptCacheStateRejectsMalformedEntries() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let key = LMOutput.Key<MLXArray>("test.state")
        var state = LMOutput.State()
        state[key] = MLXArray([Int32(1)])
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)

        // eval before overwriting file -- otherwise the lazy read picks
        // up the corruption
        eval(arrays)

        var malformedMetadata = savedMetadata
        malformedMetadata["1.__mlx_lm_state_count"] = "2"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheStateRejectsUnknownVersions() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let key = LMOutput.Key<MLXArray>("test.state")
        var state = LMOutput.State()
        state[key] = MLXArray([Int32(1)])
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)
        eval(arrays)
        var malformedMetadata = savedMetadata
        malformedMetadata["1.__mlx_lm_state_version"] = "999"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheStateRejectsDuplicateKeys() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let firstKey = LMOutput.Key<MLXArray>("test.first")
        let secondKey = LMOutput.Key<MLXArray>("test.second")
        var state = LMOutput.State()
        state[firstKey] = MLXArray([Int32(1)])
        state[secondKey] = MLXArray([Int32(2)])
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)
        eval(arrays)
        var malformedMetadata = savedMetadata
        malformedMetadata["1.__mlx_lm_state_1_key"] = "test.first"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheStateRejectsUnexpectedTensors() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let key = LMOutput.Key<MLXArray>("test.state")
        var state = LMOutput.State()
        state[key] = MLXArray([Int32(1)])
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (storedArrays, metadata) = try loadArraysAndMetadata(url: url)
        eval(storedArrays)

        var malformedArrays = storedArrays
        malformedArrays["__mlx_lm_state_tensor_99"] = MLXArray([Int32(99)])
        try save(arrays: malformedArrays, metadata: metadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheStateRequiresCompatibilityMarker() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let key = LMOutput.Key<MLXArray>("test.state")
        var state = LMOutput.State()
        state[key] = MLXArray([Int32(1)])
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (arrays, storedMetadata) = try loadArraysAndMetadata(url: url)
        eval(arrays)
        var malformedMetadata = storedMetadata
        malformedMetadata["2.0"] = "KVCache"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheCompatibilityMarkerRequiresState() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache])
        let (arrays, storedMetadata) = try loadArraysAndMetadata(url: url)
        eval(arrays)

        var malformedMetadata = storedMetadata
        malformedMetadata["2.0"] = "__mlx_lm_state_v1__:KVCache"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testQuantizedKVCacheRestoresNonDefaultQuantizationMetadata() throws {
        let cache = QuantizedKVCache(groupSize: 64, bits: 4)
        let keys = MLXArray.ones([1, 1, 4, 32], dtype: .bfloat16)
        let values = MLXArray.ones([1, 1, 4, 32], dtype: .bfloat16)
        _ = cache.updateQuantized(keys: keys, values: values)

        #expect(cache.groupSize == 32)
        #expect(cache.bits == 4)

        let url = tempURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        let restored = try #require(loaded[0] as? QuantizedKVCache)
        #expect(restored.groupSize == 32)
        #expect(restored.bits == 4)
        #expect(restored.metaState == cache.metaState)

        let moreKeys = MLXArray.zeros([1, 1, 1, 32], dtype: .bfloat16)
        let moreValues = MLXArray.zeros([1, 1, 1, 32], dtype: .bfloat16)
        _ = restored.updateQuantized(keys: moreKeys, values: moreValues)

        #expect(restored.groupSize == 32)
        #expect(restored.bits == 4)
    }

    @Test func testQuantizedKVCacheMetaStateRestoresQuantizationMetadataWithoutState() {
        let cache = QuantizedKVCache()

        cache.metaState = ["256", "11", "32", "4"]

        #expect(cache.offset == 11)
        #expect(cache.groupSize == 32)
        #expect(cache.bits == 4)
        #expect(cache.metaState == ["256", "11", "32", "4"])
    }

    @Test func testQuantizedKVCacheCopyPreservesRestoredQuantizationMetadata() throws {
        let cache = QuantizedKVCache()
        cache.metaState = ["256", "5", "32", "4"]

        let copied = try #require(cache.copy() as? QuantizedKVCache)

        #expect(copied.offset == 5)
        #expect(copied.groupSize == 32)
        #expect(copied.bits == 4)
        #expect(copied.metaState == cache.metaState)
    }

    @Test func testEmptyKVCacheSimpleToQuantizedPreservesRequestedQuantizationMetadata() throws {
        let cache = KVCacheSimple()
        cache.offset = 7

        let quantized = try cache.toQuantized(groupSize: 128, bits: 4)

        #expect(quantized.offset == 7)
        #expect(quantized.groupSize == 128)
        #expect(quantized.bits == 4)
        #expect(quantized.metaState == ["256", "7", "128", "4"])
    }

    @Test func testDirectKVCacheQuantizationFailuresAreRecoverable() throws {
        let incompatible = KVCacheSimple()
        incompatible.state = [
            MLXArray.zeros([1, 2, 4, 5], dtype: .float32, stream: .cpu),
            MLXArray.zeros([1, 2, 4, 5], dtype: .float32, stream: .cpu),
        ]

        #expect(throws: KVCacheError.self) {
            _ = try incompatible.toQuantized(groupSize: 64, bits: 4)
        }
        #expect(throws: KVCacheError.self) {
            _ = try RotatingKVCache(maxSize: 32).toQuantized()
        }
    }

    @Test func testPromptCacheRestorationRejectsIncompleteState() throws {
        try expectPromptCacheLoadToFail([
            .init(className: "KVCacheSimple", arrayCount: 1, metadata: [""]),
            .init(
                className: "RotatingKVCache", arrayCount: 1,
                metadata: ["0", "32", "256", "0", "0", "modelNative"]),
            .init(
                className: "QuantizedKVCache", arrayCount: 3,
                metadata: ["256", "0", "64", "4"]),
            .init(
                className: "VarianceNormalizedKVCache", arrayCount: 7,
                metadata: ["32", "32", "4", "4", "2", "1", "0"]),
            .init(className: "ChunkedKVCache", arrayCount: 1, metadata: ["None", "0"]),
        ])
    }

    @Test func testPromptCacheRestorationRejectsInvalidMetadata() throws {
        try expectPromptCacheLoadToFail([
            .init(className: "KVCacheSimple", arrayCount: 0, metadata: ["unexpected"]),
            .init(
                className: "RotatingKVCache", arrayCount: 0,
                metadata: ["0", "32", "invalid", "0", "0", "modelNative"]),
            .init(
                className: "RotatingKVCache", arrayCount: 0,
                metadata: ["0", "32", "256", "0", "0", "invalid-origin"]),
            .init(
                className: "QuantizedKVCache", arrayCount: 0,
                metadata: ["256", "invalid", "64", "4"]),
            .init(
                className: "VarianceNormalizedKVCache", arrayCount: 0,
                metadata: ["24", "0", "4", "4", "2", "0", "0"]),
            .init(
                className: "VarianceNormalizedKVCache", arrayCount: 8,
                metadata: ["32", "31", "4", "4", "2", "1", "0"]),
            .init(className: "ChunkedKVCache", arrayCount: 0, metadata: ["invalid", "0"]),
        ])
    }

    @Test func testPromptCacheRestorationRejectsInvalidStateRank() throws {
        try expectPromptCacheLoadToFail([
            .init(
                className: "KVCacheSimple", arrayCount: 2, metadata: [""],
                arrayShape: [1, 1]),
            .init(
                className: "VarianceNormalizedKVCache", arrayCount: 8,
                metadata: ["32", "32", "4", "4", "2", "1", "0"],
                arrayShape: [1, 1]),
        ])
    }

    @Test func testPromptCacheRestorationAcceptsLegacyRotatingMetadata() throws {
        let fixture = SerializedCacheFixture(
            className: "RotatingKVCache",
            arrayCount: 2,
            metadata: ["0", "32", "256", "1", "1"])
        let url = try writePromptCacheFixture(fixture)
        defer { try? FileManager.default.removeItem(at: url) }

        let (restored, _) = try loadPromptCache(url: url)
        let rotating = try #require(restored.first as? RotatingKVCache)

        #expect(rotating.capacityOrigin == .modelNative)
        // Legacy metadata has no wrapped flag; the restored cache derives it (an
        // unwrapped layout here) and re-serializes with the flag appended.
        #expect(rotating.metaState == ["0", "32", "256", "1", "1", "modelNative", "false"])
    }

    @Test func testPromptCacheRoundTripPreservesEmptyCaches() throws {
        let populated = KVCacheSimple()
        populated.state = [
            MLXArray.ones([1, 1, 1, 32], stream: .cpu),
            MLXArray.ones([1, 1, 1, 32], stream: .cpu),
        ]
        let caches: [KVCache] = [
            KVCacheSimple(),
            populated,
            RotatingKVCache(maxSize: 32),
            QuantizedKVCache(),
            ChunkedKVCache(chunkSize: 16),
        ]
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: caches)
        let (restored, _) = try loadPromptCache(url: url)

        #expect(restored.count == caches.count)
        #expect(restored[0] is KVCacheSimple)
        #expect(restored[0].state.isEmpty)
        #expect(restored[1].state.count == 2)
        #expect(restored[2] is RotatingKVCache)
        #expect(restored[2].state.isEmpty)
        #expect(restored[3] is QuantizedKVCache)
        #expect(restored[3].state.isEmpty)
        #expect(restored[4] is ChunkedKVCache)
        #expect(restored[4].state.isEmpty)
    }

    // MARK: - ArraysCache sparse slot round-trip

    @Test func testArraysCacheSparseSlots() throws {
        let cache = ArraysCache(size: 3)
        let a = MLXArray.ones([2, 4], dtype: .float32) * 3.0
        let b = MLXArray.ones([2, 4], dtype: .float32) * 7.0
        cache[0] = a
        // slot 1 stays nil
        cache[2] = b

        let url = tempURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? ArraysCache)
        #expect(restored.slotCount == 3)
        #expect(restored[0] != nil)
        #expect(restored[1] == nil)
        #expect(restored[2] != nil)
        #expect(allClose(restored[0]!, a).item(Bool.self))
        #expect(allClose(restored[2]!, b).item(Bool.self))
    }

    // MARK: - ArraysCache leftPadding round-trip

    @Test func testArraysCacheLeftPadding() throws {
        let cache = ArraysCache(size: 2, leftPadding: [0, 5])
        let a = MLXArray.ones([2, 4], dtype: .float32)
        let b = MLXArray.ones([2, 4], dtype: .float32) * 2.0
        cache[0] = a
        cache[1] = b

        let url = tempURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        let restored = try #require(loaded[0] as? ArraysCache)
        #expect(restored.leftPaddingValues == [0, 5])
        assertArraysClose(restored.state, cache.state)
    }

    @Test func testArraysCacheMaskUsesLeftPaddingAfterStateUpdate() throws {
        let cache = ArraysCache(size: 2, leftPadding: [1, 3])
        cache[0] = MLXArray.ones([2, 4], dtype: .float32)

        let mask = try #require(cache.makeMask(N: 4))
        #expect(
            mask.asArray(Bool.self) == [
                false, true, true, true,
                false, false, false, true,
            ])
    }

    @Test func testArraysCacheAdvanceUpdatesSequenceMetadataOnly() throws {
        let cache = ArraysCache(size: 2, leftPadding: [3, 5])
        cache.offset = 7
        cache.prepare(lengths: [4, 6])

        cache.advance(2)

        #expect(cache.offset == 7)
        #expect(cache.leftPaddingValues == [1, 3])
        #expect(cache.lengthsValues == [2, 4])
    }

    @Test func testArraysCacheMaskUsesLengthsWhenLeftPaddingIsAbsent() throws {
        let cache = ArraysCache(size: 2)
        cache.prepare(lengths: [1, 3])

        let mask = try #require(cache.makeMask(N: 4))
        #expect(
            mask.asArray(Bool.self) == [
                true, false, false, false,
                true, true, true, false,
            ])
    }

    @Test func testTextSequenceLengthsComeFromAttentionMask() throws {
        let tokens = MLXArray(0 ..< 8).reshaped(2, 4)
        let mask = MLXArray([1, 1, 0, 0, 1, 1, 1, 0]).reshaped(2, 4)
        let text = LMInput.Text(tokens: tokens, mask: mask)

        #expect(text.sequenceLengths == [2, 3])
    }

    @Test func testTextSequenceLengthsInferUniformBatches() throws {
        let text = LMInput.Text(tokens: MLXArray(0 ..< 8).reshaped(2, 4))

        #expect(text.sequenceLengths == [4, 4])
    }

    @Test func testCacheListForwardsPrepareAndFinalize() throws {
        let arrays = ArraysCache(size: 2)
        let cache = CacheList(arrays, KVCacheSimple())

        cache.prepare(lengths: [2, 4])
        #expect(arrays.lengthsValues == [2, 4])

        cache.finalize()
        #expect(arrays.lengthsValues == nil)
    }

    @Test func testCacheListForwardsLifecycleThroughKVCacheProtocol() throws {
        let lifecycle = LifecycleRecordingCache()
        let cache = CacheList(KVCacheSimple(), lifecycle)

        cache.prepare(lengths: [2, 4])
        #expect(lifecycle.preparedLengths == [2, 4])

        cache.finalize()
        #expect(lifecycle.finalizeCallCount == 1)
    }

    @Test func testWithPreparedCacheScopesSequenceMetadata() throws {
        let cache = ArraysCache(size: 2)

        withPreparedCache([cache], lengths: [2, 4]) {
            #expect(cache.lengthsValues == [2, 4])
        }

        #expect(cache.lengthsValues == nil)
    }

    @Test func testArraysCacheLengthsRoundTrip() throws {
        let cache = ArraysCache(size: 2)
        cache.prepare(lengths: [4, 2])
        cache[0] = MLXArray.ones([2, 4], dtype: .float32)

        let url = tempURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        let restored = try #require(loaded[0] as? ArraysCache)
        #expect(restored.currentLengths?.asArray(Int.self) == [4, 2])
        #expect(restored.lengthsValues == [4, 2])
        assertArraysClose(restored.state, cache.state)
    }

    @Test func testArraysCacheAdvanceUpdatesLengthsAndLeftPaddingMasks() throws {
        let cache = ArraysCache(size: 2, leftPadding: [1, 3])
        cache.prepare(lengths: [4, 2])
        cache.advance(2)

        #expect(cache.leftPaddingValues == [-1, 1])
        #expect(cache.currentLengths?.asArray(Int.self) == [2, 0])

        let mask = try #require(cache.makeMask(N: 3))
        #expect(mask.asArray(Bool.self) == [true, true, true, false, true, true])

        let lengthOnly = ArraysCache(size: 1)
        lengthOnly.prepare(lengths: [2, 0])
        let lengthMask = try #require(lengthOnly.makeMask(N: 3))
        #expect(lengthMask.asArray(Bool.self) == [true, true, false, false, false, false])

        cache.finalize()
        #expect(cache.leftPaddingValues == nil)
        #expect(cache.currentLengths == nil)
    }

    @Test func testArraysCacheFilterAndExtendPreserveBatchMetadata() throws {
        let first = ArraysCache(size: 1, leftPadding: [0, 2])
        first.prepare(lengths: [5, 3])
        first[0] = MLXArray.ones([2, 2], dtype: .float32)

        first.filter(batchIndices: MLXArray([1]))
        #expect(first.leftPaddingValues == [2])
        #expect(first.currentLengths?.asArray(Int.self) == [3])
        #expect(first[0]?.shape == [1, 2])

        let second = ArraysCache(size: 1, leftPadding: [1, 4])
        second.prepare(lengths: [6, 2])
        second[0] = MLXArray.ones([2, 2], dtype: .float32) * 2

        first.extend(other: second)
        #expect(first.leftPaddingValues == [2, 1, 4])
        #expect(first.currentLengths?.asArray(Int.self) == [3, 6, 2])
        #expect(first[0]?.shape == [3, 2])
    }

    @Test func testAttentionMaskUsesSharedCausalCachePath() throws {
        let cache = KVCacheSimple()
        let prefillInput = MLXArray.ones([1, 3, 8], dtype: .float32)

        let prefillMask = createAttentionMask(h: prefillInput, cache: cache)
        if case .causal = prefillMask {
            // Expected for multi-token prefill: Falcon H1 uses the shared symbolic causal mask path.
        } else {
            Issue.record("Expected symbolic causal attention mask for prefill")
        }

        let tokenInput = MLXArray.ones([1, 1, 8], dtype: .float32)
        let tokenMask = createAttentionMask(h: tokenInput, cache: cache)
        if case .none = tokenMask {
            // Expected for one-token decode: no materialized mask is needed.
        } else {
            Issue.record("Expected no attention mask for one-token decode")
        }

        cache.offset = 2
        let forcedMask = createAttentionMask(h: prefillInput, cache: cache, returnArray: true)
        guard case .array(let mask) = forcedMask else {
            Issue.record("Expected forced attention mask array")
            return
        }
        #expect(mask.shape == [3, 5])
        #expect(
            mask.asArray(Bool.self) == [
                true, true, true, false, false,
                true, true, true, true, false,
                true, true, true, true, true,
            ])
    }

    @Test func testSSMMaskUsesSharedMambaMetadataPath() throws {
        let leftPadded = MambaCache(leftPadding: [1, 3])
        let input = MLXArray.ones([2, 4, 8], dtype: .float32)

        let leftPaddingMask = try #require(createSSMMask(h: input, cache: leftPadded))
        #expect(
            leftPaddingMask.asArray(Bool.self) == [
                false, true, true, true,
                false, false, false, true,
            ])

        let lengthMasked = MambaCache()
        lengthMasked.prepare(lengths: [3, 1])
        let lengthsMask = try #require(createSSMMask(h: input, cache: lengthMasked))
        #expect(
            lengthsMask.asArray(Bool.self) == [
                true, true, true, false,
                true, false, false, false,
            ])
    }

    @Test func testCacheListPrepareFinalizePropagatesThroughNestedHybridCaches() throws {
        let mamba = MambaCache(leftPadding: [0, 2])
        let arrays = ArraysCache(size: 1)
        let nested = CacheList(CacheList(mamba), arrays)

        nested.prepare(lengths: [4, 1])

        #expect(mamba.currentLengths?.asArray(Int.self) == [4, 1])
        #expect(arrays.currentLengths?.asArray(Int.self) == [4, 1])

        nested.finalize()

        #expect(mamba.currentLengths == nil)
        #expect(mamba.leftPaddingValues == nil)
        #expect(arrays.currentLengths == nil)
    }

    @Test func testMambaCacheCopyPreservesBatchMaskMetadata() throws {
        let cache = MambaCache(leftPadding: [2, 0])
        cache.prepare(lengths: [5, 3])
        cache[0] = MLXArray.ones([2, 3, 4], dtype: .float32)
        cache[1] = MLXArray.ones([2, 1, 4, 4], dtype: .float32)

        let copied = try #require(cache.copy() as? MambaCache)

        #expect(copied.leftPaddingValues == [2, 0])
        #expect(copied.currentLengths?.asArray(Int.self) == [5, 3])
        #expect(copied[0]?.shape == [2, 3, 4])
        #expect(copied[1]?.shape == [2, 1, 4, 4])
    }

    @Test func testArraysCacheFilterKeepsSequenceMetadata() throws {
        let cache = ArraysCache(size: 2, leftPadding: [1, 3])
        cache.prepare(lengths: [2, 4])
        cache[0] = MLXArray.ones([2, 4], dtype: .float32)

        cache.filter(batchIndices: MLXArray([1]))

        #expect(cache.leftPaddingValues == [3])
        #expect(cache.lengthsValues == [4])
    }

    @Test func testArraysCacheExtendPadsMissingSlotsAndMetadata() throws {
        let first = ArraysCache(size: 2, leftPadding: [1, 3])
        first.prepare(lengths: [2, 4])
        first[0] = MLXArray.ones([2, 4], dtype: .float32)

        let second = ArraysCache(size: 2)
        second[1] = MLXArray.ones([1, 4], dtype: .float32) * 2

        first.extend(other: second)

        #expect(first[0]?.shape == [3, 4])
        #expect(first[1]?.shape == [3, 4])
        #expect(first.leftPaddingValues == [1, 3, 0])
        #expect(first.lengthsValues == [2, 4, 0])
    }

    @Test func testArraysCacheCopyPreservesSparseSlotsAndMetadata() throws {
        let cache = ArraysCache(size: 3, leftPadding: [2])
        cache.prepare(lengths: [5])
        cache[2] = MLXArray.ones([1, 4], dtype: .float32)

        let copied = try #require(cache.copy() as? ArraysCache)

        #expect(copied.slotCount == 3)
        #expect(copied[0] == nil)
        #expect(copied[1] == nil)
        #expect(copied[2] != nil)
        #expect(copied.leftPaddingValues == [2])
        #expect(copied.lengthsValues == [5])
    }

    // MARK: - MambaCache type preservation

    @Test func testMambaCacheRoundTrip() throws {
        let cache = MambaCache()
        let a = MLXArray.ones([2, 4], dtype: .float32) * 5.0
        let b = MLXArray.ones([2, 4], dtype: .float32) * 9.0
        cache[0] = a
        cache[1] = b

        let url = tempURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? MambaCache)
        #expect(restored.slotCount == 2)
        assertArraysClose(restored.state, cache.state)
    }

    // MARK: - CacheList with KV caches

    @Test func testCacheListKVCaches() throws {
        let simple = KVCacheSimple()
        let rotating = RotatingKVCache(maxSize: 32)

        let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        _ = simple.update(keys: keys, values: values)
        _ = rotating.update(keys: keys * 2.0, values: values * 2.0)

        let cacheList = CacheList(simple, rotating)

        let url = tempURL()
        try savePromptCache(url: url, cache: [cacheList], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? CacheList)
        let child0 = try #require(restored[0] as? KVCacheSimple)
        let child1 = try #require(restored[1] as? RotatingKVCache)

        assertArraysClose(child0.state, simple.state, label: "child0")
        assertArraysClose(child1.state, rotating.state, label: "child1")
        #expect(child1.metaState == rotating.metaState)
    }

    // MARK: - CacheList with hybrid (MambaCache + KVCacheSimple)

    @Test func testCacheListHybrid() throws {
        let mamba = MambaCache()
        mamba[0] = MLXArray.ones([2, 4], dtype: .float32) * 3.0
        mamba[1] = MLXArray.ones([2, 4], dtype: .float32) * 4.0

        let simple = KVCacheSimple()
        let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        _ = simple.update(keys: keys, values: values)

        let cacheList = CacheList(mamba, simple)

        let url = tempURL()
        try savePromptCache(url: url, cache: [cacheList], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? CacheList)
        let restoredMamba = try #require(restored[0] as? MambaCache)
        let restoredSimple = try #require(restored[1] as? KVCacheSimple)

        assertArraysClose(restoredMamba.state, mamba.state, label: "mamba")
        assertArraysClose(restoredSimple.state, simple.state, label: "simple")
    }

    // MARK: - Simple cache round-trip with value assertions

    @Test func testSimpleCacheRoundTrip() throws {
        let cache = KVCacheSimple()
        let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        _ = cache.update(keys: keys, values: values)

        let url = tempURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)
        #expect(loaded.count == 1)
        assertArraysClose(loaded[0].state, cache.state)
    }

    // MARK: - ArraysCache fully populated round-trip

    @Test func testArraysCacheFullyPopulated() throws {
        let cache = ArraysCache(size: 2)
        cache[0] = MLXArray.ones([2, 4], dtype: .float32)
        cache[1] = MLXArray.ones([2, 4], dtype: .float32) * 2.0

        let url = tempURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? ArraysCache)
        #expect(restored.slotCount == 2)
        assertArraysClose(restored.state, cache.state)
    }

    /// Verify that copy() produces an independent cache: same type, same state,
    /// but mutating the copy does not affect the original.
    @Test(
        .serialized,
        arguments: cacheCreators)
    func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
        let original = creator()

        let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
        let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)

        // populate the original
        switch original {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        case let quantized as QuantizedKVCache:
            _ = quantized.updateQuantized(keys: keys, values: values)
        default:
            _ = original.update(keys: keys, values: values)
        }

        let originalOffset = original.offset
        let originalState = original.state
        eval(originalState)
        let originalMeta = original.metaState

        // copy
        let copied = original.copy()

        // same type
        #expect(type(of: original) == type(of: copied))

        // same offset and metadata
        #expect(copied.offset == originalOffset)
        #expect(copied.metaState == originalMeta)

        // same state values
        let copiedState = copied.state
        eval(copiedState)
        #expect(copiedState.count == originalState.count)
        for (origArr, copyArr) in zip(originalState, copiedState) {
            #expect(origArr.shape == copyArr.shape)
            #expect(allClose(origArr, copyArr).item(Bool.self))
        }

        // mutate the copy — push more tokens through it
        let moreKeys = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
        let moreValues = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)

        switch copied {
        case let arrays as ArraysCache:
            // overwrite slot 0 with a different array
            arrays[0] = moreKeys
        case let quantized as QuantizedKVCache:
            _ = quantized.updateQuantized(keys: moreKeys, values: moreValues)
        default:
            _ = copied.update(keys: moreKeys, values: moreValues)
        }

        // original must be unchanged
        #expect(original.offset == originalOffset)
        #expect(original.metaState == originalMeta)
        let currentState = original.state
        eval(currentState)
        #expect(currentState.count == originalState.count)
        for (origArr, savedArr) in zip(currentState, originalState) {
            #expect(origArr.shape == savedArr.shape)
            #expect(allClose(origArr, savedArr).item(Bool.self))
        }
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        #expect(lhs.state.count == rhs.state.count)
    }
}

/// Verify that copy() produces an independent cache: same type, same state,
/// but mutating the copy does not affect the original.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
    let original = creator()

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)

    // populate the original
    switch original {
    case let arrays as ArraysCache:
        arrays[0] = keys
        arrays[1] = values
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: keys, values: values)
    default:
        _ = original.update(keys: keys, values: values)
    }

    // MARK: - ropeOffset overridability

    /// A `BaseKVCache` subclass reporting a per-row RoPE offset, as a batched cache does.
    private final class BatchOffsetProbeCache: BaseKVCache {
        override var ropeOffset: RoPEOffset { .batch(MLXArray([10, 20])) }

        override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            (keys, values)
        }
    }

    /// Models read `ropeOffset` through a `KVCache` reference, so a subclass override has to
    /// survive that dispatch. When `BaseKVCache` did not declare `ropeOffset`, the extension
    /// default was the witness and this resolved to `.scalar(0)`, silently ignoring the subclass.
    @Test func testSubclassRopeOffsetOverrideIsHonoredThroughKVCacheReference() {
        let cache: any KVCache = BatchOffsetProbeCache()

        guard case .batch(let offsets) = cache.ropeOffset else {
            Issue.record(
                "subclass ropeOffset override ignored — resolved to the .scalar extension default")
            return
        }
        #expect(offsets.asArray(Int32.self) == [10, 20])
    }

    // MARK: - RotatingKVCache.logicalView
    //
    // `logicalView(tail:)` is the read-only counterpart to `update(keys:values:)`: it exposes the
    // ring's contents in chronological order without writing. Speculative decoding needs it to
    // present committed history alongside K/V it has not committed yet.
    //
    // Nothing in this repo drove a `RotatingKVCache` ring past its wrap before these tests, so the
    // rotation in `updateInPlace`, the linearization in `temporalOrder`, and the windowed mask that
    // `makeMask` builds over the multi-token presentation were all uncovered. They are the
    // foundation the view rests on, so they are pinned here too.

    /// K/V whose every element encodes its own sequence position: keys hold `+p`, values `-p`.
    /// Any reordering, duplication, or K/V mix-up changes the numbers.
    private func positionedKV(
        _ positions: Range<Int>, headDim: Int = 2
    ) -> (MLXArray, MLXArray) {
        let count = positions.count
        let keys = MLXArray(
            positions.flatMap { Array(repeating: Float($0), count: headDim) },
            [1, 1, count, headDim])
        let values = MLXArray(
            positions.flatMap { Array(repeating: Float(-$0), count: headDim) },
            [1, 1, count, headDim])
        return (keys, values)
    }

    /// Recover the sequence positions encoded by `positionedKV` from a `[B, H, S, D]` key array.
    private func encodedPositions(_ keys: MLXArray) -> [Int] {
        keys[0, 0, 0..., 0].asArray(Float.self).map { Int($0) }
    }
}
}
}

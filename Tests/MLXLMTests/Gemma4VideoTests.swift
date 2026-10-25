import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

extension MLXTestingSuite {
    @Suite
    struct Gemma4VideoTests {
        @Test("Gemma consumes video pixels across vision chunks", arguments: [true, false])
        func videoPixelsChangeLogitsAcrossChunks(tieWordEmbeddings: Bool) throws {
            try withMemoryLimit {
                let model = Gemma4(try makeConfiguration(tieWordEmbeddings: tieWordEmbeddings))
                eval(model)
                let frameCount = 5
                let pixels =
                    MLXArray(0 ..< frameCount * 48)
                    .reshaped([frameCount, 3, 4, 4]).asType(.float32) / 240
                var tokens: [Int32] = [1]
                tokens.append(contentsOf: Array(repeating: 61, count: frameCount * 4))
                tokens.append(2)

                let original = try logits(model, tokens: tokens, video: pixels)
                let changedPixels = concatenated([
                    pixels[..<(frameCount - 1)],
                    MLXArray.zeros([1, 3, 4, 4]),
                ])
                let changed = try logits(model, tokens: tokens, video: changedPixels)

                #expect(original.shape == [1, 1, 64])
                #expect(original.nbytes == 64 * MemoryLayout<Float>.size)
                #expect(isFinite(original).all().item(Bool.self))
                #expect(isFinite(changed).all().item(Bool.self))
                #expect(!allClose(original, changed, rtol: 1e-6, atol: 1e-6).item(Bool.self))
            }
        }

        @Test("Gemma encodes one video frame like one image", arguments: [true, false])
        func videoFrameMatchesImage(tieWordEmbeddings: Bool) throws {
            try withMemoryLimit {
                let model = Gemma4(try makeConfiguration(tieWordEmbeddings: tieWordEmbeddings))
                eval(model)
                let pixels = MLXArray(0 ..< 48).reshaped([1, 3, 4, 4]).asType(.float32) / 48
                let videoLogits = try logits(
                    model, tokens: [1, 61, 61, 61, 61, 2], video: pixels)
                let imageInput = LMInput(
                    text: .init(tokens: MLXArray([Int32(1), 60, 60, 60, 60, 2], [1, 6])),
                    image: .init(pixels: pixels))
                let imageResult = try model.prepare(imageInput, cache: [], windowSize: nil)
                guard case .logits(let output) = imageResult else {
                    Issue.record("Gemma did not return image logits.")
                    return
                }
                eval(output.logits)

                #expect(output.logits.shape == [1, 1, 64])
                #expect(
                    allClose(videoLogits, output.logits, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            }
        }

        @Test("Gemma text callers retain every position", arguments: [true, false])
        func textForwardRetainsEveryPosition(tieWordEmbeddings: Bool) throws {
            try withMemoryLimit {
                let model = Gemma4(try makeConfiguration(tieWordEmbeddings: tieWordEmbeddings))
                eval(model)
                let tokens = MLXArray([Int32(1), 2, 3, 4], [1, 4])
                let full = model(tokens, cache: nil)
                let single = model(MLXArray([Int32(1)], [1, 1]), cache: nil)
                let prepared = try model.prepare(
                    LMInput(tokens: tokens), cache: [], windowSize: nil)
                guard case .logits(let output) = prepared else {
                    Issue.record("Gemma did not return text logits.")
                    return
                }
                eval(full, single, output.logits)

                #expect(full.shape == [1, 4, 64])
                #expect(single.shape == [1, 1, 64])
                #expect(output.logits.shape == full.shape)
                #expect(allClose(full, output.logits).item(Bool.self))
            }
        }

        @Test("Gemma multimodal preparation reduces vocabulary allocation")
        func multimodalPreparationReducesVocabularyAllocation() throws {
            try withMemoryLimit {
                let model = Gemma4(try makeConfiguration(vocabularySize: 8192))
                eval(model)
                var tokenValues = Array(repeating: Int32(1), count: 123)
                tokenValues.append(contentsOf: Array(repeating: 61, count: 4))
                tokenValues.append(2)
                let tokens = MLXArray(tokenValues, [1, tokenValues.count])
                let pixels = MLXArray.ones([1, 3, 4, 4])
                eval(tokens, pixels)

                let full = measuredAllocation { model(tokens, cache: nil) }
                let prepared = try measuredAllocation {
                    try logits(model, tokens: tokenValues, video: pixels)
                }

                #expect(full.shape == [1, 128, 8192])
                #expect(prepared.shape == [1, 1, 8192])
                #expect(full.bytes == 128 * prepared.bytes)
                #expect(prepared.peakIncrease < full.peakIncrease)
            }
        }

        @Test("Gemma rejects video pixels without matching prompt tokens")
        func rejectsVideoTokenCountMismatch() throws {
            try withMemoryLimit {
                let model = Gemma4(try makeConfiguration())
                eval(model)
                let pixels = MLXArray.ones([1, 3, 4, 4])
                let input = LMInput(
                    text: .init(tokens: MLXArray([Int32(1), 61, 2], [1, 3])),
                    video: .init(pixels: pixels))

                #expect(throws: (any Error).self) {
                    try model.prepare(input, cache: [], windowSize: nil)
                }
            }
        }

        private func logits(_ model: Gemma4, tokens: [Int32], video: MLXArray) throws -> MLXArray {
            let input = LMInput(
                text: .init(tokens: MLXArray(tokens, [1, tokens.count])),
                video: .init(pixels: video))
            let cache = model.newCache(parameters: nil)
            let result = try model.prepare(input, cache: cache, windowSize: nil)
            guard case .logits(let output) = result else {
                throw CocoaError(.coderReadCorrupt)
            }
            eval(output.logits)
            #expect(cache.allSatisfy { $0.offset == tokens.count })
            return output.logits
        }

        private func measuredAllocation(_ body: () throws -> MLXArray) rethrows -> (
            shape: [Int], bytes: Int, peakIncrease: Int
        ) {
            Memory.clearCache()
            let baseline = Memory.activeMemory
            Memory.peakMemory = 0
            let output = try body()
            eval(output)
            return (output.shape, output.nbytes, Memory.peakMemory - baseline)
        }

        private func withMemoryLimit(_ body: () throws -> Void) rethrows {
            let previousMemoryLimit = Memory.memoryLimit
            let previousCacheLimit = Memory.cacheLimit
            Memory.memoryLimit = 64 * 1024 * 1024
            Memory.cacheLimit = 0
            defer {
                Memory.clearCache()
                Memory.cacheLimit = previousCacheLimit
                Memory.memoryLimit = previousMemoryLimit
            }
            try body()
        }

        private func makeConfiguration(
            tieWordEmbeddings: Bool = true, vocabularySize: Int = 64
        ) throws -> Gemma4Configuration {
            let json = """
                {
                  "model_type": "gemma4",
                  "image_token_id": 60,
                  "video_token_id": 61,
                  "vision_soft_tokens_per_video_frame": 4,
                  "video_frame_chunk_size": 2,
                  "text_config": {
                    "hidden_size": 32,
                    "num_hidden_layers": 1,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "vocab_size": \(vocabularySize),
                    "vocab_size_per_layer_input": 64,
                    "hidden_size_per_layer_input": 8,
                    "num_kv_shared_layers": 0,
                    "sliding_window": 64,
                    "sliding_window_pattern": 2,
                    "max_position_embeddings": 128,
                    "use_double_wide_mlp": false,
                    "tie_word_embeddings": \(tieWordEmbeddings)
                  },
                  "vision_config": {
                    "num_hidden_layers": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_attention_heads": 2,
                    "head_dim": 8,
                    "patch_size": 2,
                    "default_output_length": 4,
                    "pooling_kernel_size": 1,
                    "position_embedding_size": 4
                  }
                }
                """
            return try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
        }
    }
}

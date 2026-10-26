import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
import Tokenizers

extension MLXTestingSuite {
    @Suite
    struct Gemma4VideoGeometryTests {
        @Test(arguments: [false, true])
        func defaultGeometryUsesEveryValidFeature(rectangular: Bool) async throws {
            try await withMemoryLimit {
                let (processor, tokenizer) = try await makeProcessor()
                let sourceSize =
                    rectangular
                    ? CGSize(width: 1920, height: 1080) : CGSize(width: 384, height: 384)
                let expectedSize =
                    rectangular
                    ? CGSize(width: 528, height: 288) : CGSize(width: 384, height: 384)
                let count = rectangular ? 66 : 64
                let frame = makeFrame(size: sourceSize, index: 0)
                let input = try await processor.prepare(
                    input: UserInput(
                        chat: [.user("Describe the clip.", videos: [.frames([frame])])]))
                let video = try #require(input.video)
                let geometry = try #require(video.frames)
                #expect(
                    video.pixels.shape == [1, 3, Int(expectedSize.height), Int(expectedSize.width)])
                #expect(geometry.count == 1)
                #expect(geometry[0].h == Int(expectedSize.height))
                #expect(geometry[0].w == Int(expectedSize.width))
                let tokens = input.text.tokens.asArray(Int.self)
                #expect(tokens.filter { $0 == 258_884 }.count == count)
                #expect(tokens.filter { $0 == 255_999 }.count == 1)
                #expect(tokens.filter { $0 == 258_882 }.count == 1)
                let block =
                    "00:00.000000 <|image>"
                    + String(repeating: "<|video|>", count: count) + "<image|> Describe the clip."
                #expect(tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false) == block)

                let model = try makeModel()
                let original = try logits(model: model, input: input)
                let changed = try logits(
                    model: model,
                    input: LMInput(
                        text: input.text,
                        video: .init(pixels: MLXArray.zeros(like: video.pixels), frames: geometry)))
                #expect(original.shape == [1, 1, 262_144])
                #expect(isFinite(original).all().item(Bool.self))
                #expect(!allClose(original, changed, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            }
        }

        @Test
        func mixedGeometryPreservesFrameOrderAndExcludesPadding() async throws {
            try await withMemoryLimit {
                let (processor, tokenizer) = try await makeProcessor()
                let sizes = [
                    CGSize(width: 384, height: 384), CGSize(width: 1920, height: 1080),
                    CGSize(width: 1080, height: 1920),
                ]
                let frames = sizes.enumerated().map {
                    makeFrame(size: $0.element, index: $0.offset)
                }
                let input = try await processor.prepare(
                    input: UserInput(chat: [.user("Describe the clip.", videos: [.frames(frames)])])
                )
                let video = try #require(input.video)
                let geometry = try #require(video.frames)
                #expect(video.pixels.shape == [3, 3, 528, 528])
                #expect(geometry.map(\.h) == [384, 288, 528])
                #expect(geometry.map(\.w) == [384, 528, 288])
                let counts = [64, 66, 66]
                let timestamps = ["00:00.000000", "00:00.062500", "00:00.125000"]
                let blocks = zip(timestamps, counts).map { timestamp, count in
                    timestamp + " <|image>" + String(repeating: "<|video|>", count: count)
                        + "<image|>"
                }
                let decoded = tokenizer.decode(
                    tokenIds: input.text.tokens.asArray(Int.self), skipSpecialTokens: false)
                #expect(decoded == blocks.joined(separator: " ") + " Describe the clip.")

                let changedPadding = geometry.enumerated().map { index, frame in
                    padded(
                        video.pixels[index ..< (index + 1), 0..., ..<frame.h, ..<frame.w],
                        widths: [0, 0, .init((0, 528 - frame.h)), .init((0, 528 - frame.w))],
                        value: MLXArray(1))
                }
                let model = try makeModel()
                let original = try logits(model: model, input: input)
                let changed = try logits(
                    model: model,
                    input: LMInput(
                        text: input.text,
                        video: .init(pixels: concatenated(changedPadding), frames: geometry)))
                #expect(allClose(original, changed, rtol: 0, atol: 0).item(Bool.self))
            }
        }

        @Test
        func compatibleLegacyOverridePreservesFortyFrames() async throws {
            try await withMemoryLimit {
                let (processor, tokenizer) = try await makeProcessor(legacyGeometry: true)
                let frames = (0 ..< 40).map { index in
                    makeFrame(size: CGSize(width: 4, height: 4), index: index)
                }
                let input = try await processor.prepare(
                    input: UserInput(chat: [.user("Describe the clip.", videos: [.frames(frames)])])
                )
                let video = try #require(input.video)
                let geometry = try #require(video.frames)
                #expect(video.pixels.shape == [40, 3, 4, 4])
                #expect(geometry.count == 40)
                let tokens = input.text.tokens.asArray(Int.self)
                #expect(tokens.filter { $0 == 258_884 }.count == 160)
                #expect(tokens.filter { $0 == 255_999 }.count == 40)
                #expect(tokens.filter { $0 == 258_882 }.count == 40)
                let decoded = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
                #expect(decoded.contains("00:00.062500 <|image>"))
                #expect(decoded.contains("00:02.437500 <|image>"))
                let model = try makeModel(patchSize: 2, poolingKernelSize: 1)
                let output = try logits(model: model, input: input)
                #expect(output.shape == [1, 1, 262_144])
                #expect(isFinite(output).all().item(Bool.self))
            }
        }

        private func makeProcessor(legacyGeometry: Bool = false) async throws -> (
            Gemma4Processor, any MLXLMCommon.Tokenizer
        ) {
            let directory = try #require(
                Bundle.module.url(forResource: "GemmaVideoTokenizer", withExtension: nil))
            let upstream = try await AutoTokenizer.from(modelFolder: directory)
            let tokenizer = #adaptHuggingFaceTokenizer(upstream)
            let data: Data
            if legacyGeometry {
                data = Data(
                    """
                    {"processor_class": "Gemma4Processor", "patch_size": 2,
                     "pooling_kernel_size": 1, "video_seq_length": 4,
                     "video_frame_size": {"width": 4, "height": 4},
                     "video_max_frames": 2, "video_fps": 1}
                    """.utf8)
            } else {
                data = try Data(
                    contentsOf: directory.appendingPathComponent("preprocessor_config.json"))
            }
            let configuration = try JSONDecoder().decode(
                Gemma4ProcessorConfiguration.self, from: data)
            return (Gemma4Processor(configuration, tokenizer: tokenizer), tokenizer)
        }

        private func makeFrame(size: CGSize, index: Int) -> VideoFrame {
            VideoFrame(
                image: .ciImage(
                    CIImage(
                        color: CIColor(red: index == 0 ? 1 : 0, green: index == 1 ? 1 : 0, blue: 1)
                    )
                    .cropped(to: CGRect(origin: .zero, size: size))),
                timeStamp: CMTime(value: Int64(index), timescale: 16))
        }

        private func logits(model: Gemma4, input: LMInput) throws -> MLXArray {
            let result = try model.prepare(
                input, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(chunking: .unchunked))
            guard case .logits(let output) = result else {
                throw CocoaError(.coderReadCorrupt)
            }
            eval(output.logits)
            return output.logits
        }

        private func makeModel(patchSize: Int = 16, poolingKernelSize: Int = 3) throws -> Gemma4 {
            let configuration = try JSONDecoder().decode(
                Gemma4Configuration.self,
                from: Data(
                    """
                    {
                      "model_type": "gemma4", "video_token_id": 258884,
                      "video_frame_chunk_size": 1,
                      "text_config": {
                        "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
                        "num_attention_heads": 1, "num_key_value_heads": 1,
                        "head_dim": 8, "global_head_dim": 8, "vocab_size": 262144,
                        "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0,
                        "sliding_window": 512, "sliding_window_pattern": 2,
                        "max_position_embeddings": 512, "use_double_wide_mlp": false,
                        "tie_word_embeddings": true
                      },
                      "vision_config": {
                        "num_hidden_layers": 1, "hidden_size": 8, "intermediate_size": 16,
                        "num_attention_heads": 1, "head_dim": 8,
                        "patch_size": \(patchSize), "pooling_kernel_size": \(poolingKernelSize),
                        "position_embedding_size": 64
                      }
                    }
                    """.utf8))
            let model = Gemma4(configuration)
            model.apply { $0.asType(.float16) }
            eval(model)
            return model
        }

        private func withMemoryLimit(_ body: () async throws -> Void) async rethrows {
            let previousMemoryLimit = Memory.memoryLimit
            let previousCacheLimit = Memory.cacheLimit
            Memory.memoryLimit = 64 * 1024 * 1024
            Memory.cacheLimit = 0
            defer {
                Stream.defaultStream.synchronize()
                Memory.clearCache()
                Memory.cacheLimit = previousCacheLimit
                Memory.memoryLimit = previousMemoryLimit
            }
            try await body()
        }
    }
}

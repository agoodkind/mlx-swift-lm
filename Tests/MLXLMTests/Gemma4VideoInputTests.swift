// Copyright © 2026 Apple Inc.

import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

// Video frames use the image vision tower with a per-frame maximum.

struct Gemma4VideoInputTests {

    /// `Gemma4Configuration` decodes `video_token_id` when present and leaves it nil
    /// otherwise (older text+vision checkpoints keep working unchanged).
    @Test("Gemma4Configuration decodes video_token_id")
    func decodesVideoTokenId() throws {
        let withVideo = try Self.decodeConfig(videoTokenLine: "\"video_token_id\": 258884,")
        #expect(withVideo.videoTokenId == 258884)
        #expect(withVideo.visionSoftTokensPerVideoFrame == 70)

        let withoutVideo = try Self.decodeConfig(videoTokenLine: "")
        #expect(withoutVideo.videoTokenId == nil)
    }

    /// Image and video input use every feature from the shared vision tower.
    @Test("Image and video pixels scatter identically (no truncation)")
    func imageAndVideoAgree() throws {
        let model = Gemma4(try Self.makeTinyConfig(videoSoftTokens: 4))
        eval(model)

        let imageTokenId = 100
        let videoTokenId = 101
        func prompt(_ mm: Int) -> [Int] { [5, 6, mm, mm, mm, mm, 7, 8] }
        // 8x8 with patch_size 4 → a 2x2 patch grid → 4 soft tokens per image
        // under the aspect-preserving tower (which emits patch-count tokens).
        let pixels = (MLXArray(0 ..< 192).reshaped([1, 3, 8, 8]).asType(.float32)) / 192.0

        func lastLogits(mm: Int, asVideo: Bool) throws -> MLXArray {
            let tokens = MLXArray(prompt(mm).map { Int32($0) }).expandedDimensions(axis: 0)
            let text = LMInput.Text(tokens: tokens)
            let input =
                asVideo
                ? LMInput(text: text, video: LMInput.ProcessedVideo(pixels: pixels))
                : LMInput(text: text, image: LMInput.ProcessedImage(pixels: pixels))
            let result = try model.prepare(
                input, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 1024))
            guard case .logits(let out) = result else {
                Issue.record("Expected .logits from Gemma4.prepare (multimodal branch)")
                return MLXArray(0)
            }
            let logits = out.logits[0..., -1, 0...]
            eval(logits)
            return logits
        }

        let imageLogits = try lastLogits(mm: imageTokenId, asVideo: false)
        let videoLogits = try lastLogits(mm: videoTokenId, asVideo: true)
        #expect(imageLogits.shape == [1, 200])
        #expect(
            allClose(imageLogits, videoLogits, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "Video pixels must scatter through the vision tower identically to image pixels.")
    }

    /// Reject frames that require more features than the configured maximum.
    @Test("Multi-frame video rejects geometry above the per-frame maximum")
    func videoRejectsFramesAboveBudget() throws {
        let softPerFrame = 2
        let numFrames = 2
        let videoTokenId = 101
        let model = Gemma4(try Self.makeTinyConfig(videoSoftTokens: softPerFrame))
        eval(model)

        var tokens: [Int32] = [5, 6]
        tokens += Array(repeating: Int32(videoTokenId), count: numFrames * softPerFrame)
        tokens += [7]
        let text = LMInput.Text(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        let pixels =
            (MLXArray(0 ..< (numFrames * 3 * 8 * 8)).reshaped([numFrames, 3, 8, 8])
                .asType(.float32)) / 400.0
        let input = LMInput(text: text, video: LMInput.ProcessedVideo(pixels: pixels))

        #expect(throws: (any Error).self) {
            try model.prepare(
                input, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 1024))
        }
    }

    // MARK: - Helpers

    private static func makeTinyConfig(videoSoftTokens: Int) throws -> Gemma4Configuration {
        try decodeConfig(
            videoTokenLine: "\"video_token_id\": 101,",
            extraLines: "\"vision_soft_tokens_per_video_frame\": \(videoSoftTokens),")
    }

    private static func decodeConfig(videoTokenLine: String, extraLines: String = "") throws
        -> Gemma4Configuration
    {
        let json = """
            {
              "model_type": "gemma4",
              "image_token_id": 100,
              \(videoTokenLine)
              \(extraLines)
              "text_config": {
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                "global_head_dim": 32, "vocab_size": 200, "vocab_size_per_layer_input": 200,
                "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 8,
                "sliding_window": 16, "sliding_window_pattern": 2, "max_position_embeddings": 512
              },
              "vision_config": {
                "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_attention_heads": 2, "head_dim": 8, "patch_size": 4,
                "default_output_length": 4, "pooling_kernel_size": 1, "position_embedding_size": 8
              }
            }
            """
        return try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
    }

}

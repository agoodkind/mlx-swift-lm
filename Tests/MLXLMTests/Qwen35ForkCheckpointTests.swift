import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

extension MLXTestingSuite {
    @Suite(.serialized)
    struct Qwen35ForkCheckpointTests {
        @Test func realCheckpointLoadingHonorsEmbeddedMTPSelection() throws {
            let previousMTP = ProcessInfo.processInfo.environment["SWIFTLM_MTP_ENABLE"]
            let previousMemoryLimit = Memory.memoryLimit
            let previousCacheLimit = Memory.cacheLimit
            Memory.memoryLimit = 64 * 1024 * 1024
            Memory.cacheLimit = 0
            defer {
                if let previousMTP {
                    setenv("SWIFTLM_MTP_ENABLE", previousMTP, 1)
                } else {
                    unsetenv("SWIFTLM_MTP_ENABLE")
                }
                Memory.clearCache()
                Memory.cacheLimit = previousCacheLimit
                Memory.memoryLimit = previousMemoryLimit
            }

            let json = """
                {
                  "model_type": "qwen3_5_text",
                  "hidden_size": 16, "num_hidden_layers": 1,
                  "intermediate_size": 32, "num_attention_heads": 2,
                  "num_key_value_heads": 1, "head_dim": 8,
                  "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                  "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                  "linear_conv_kernel_dim": 2, "vocab_size": 16,
                  "rms_norm_eps": 1e-6, "full_attention_interval": 1,
                  "tie_word_embeddings": true, "mtp_num_hidden_layers": 1
                }
                """
            let configuration = try JSONDecoder().decode(
                MLXLLM.Qwen35TextConfiguration.self, from: Data(json.utf8))
            #expect(configuration.numNextnPredictLayers == 1)

            setenv("SWIFTLM_MTP_ENABLE", "1", 1)
            let original = MLXLLM.Qwen35TextModel(configuration)
            let expected = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
            #expect(original.mtp.count == 1)
            let fixture = Dictionary(
                uniqueKeysWithValues: expected.map { name, array in
                    (name.replacingOccurrences(of: "mtp.0.", with: "mtp."), array)
                })
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            try save(arrays: fixture, url: directory.appendingPathComponent("model.safetensors"))

            for retained in [true, false] {
                setenv("SWIFTLM_MTP_ENABLE", retained ? "1" : "0", 1)
                let model = MLXLLM.Qwen35TextModel(configuration)
                try loadWeights(modelDirectory: directory, model: model)
                let loaded = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

                #expect(model.mtp.count == (retained ? 1 : 0))
                for (name, array) in expected {
                    if !retained && name.hasPrefix("mtp.") {
                        #expect(loaded[name] == nil)
                    } else {
                        let actual = try #require(loaded[name])
                        #expect(arrayEqual(actual, array).item(Bool.self))
                    }
                }
            }
        }
    }
}

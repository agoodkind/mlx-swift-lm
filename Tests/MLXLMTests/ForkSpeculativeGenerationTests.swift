import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

extension MLXTestingSuite {
    @Suite
    struct ForkSpeculativeGenerationTests {
        @Test(arguments: [false, true])
        func qwenMTPMatchesGreedy(acceptDrafts: Bool) throws {
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

            let configuration = try JSONDecoder().decode(
                Qwen35TextConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "qwen3_5_text",
                      "hidden_size": 16,
                      "num_hidden_layers": 1,
                      "intermediate_size": 32,
                      "num_attention_heads": 2,
                      "num_key_value_heads": 1,
                      "head_dim": 8,
                      "linear_num_value_heads": 2,
                      "linear_num_key_heads": 1,
                      "linear_key_head_dim": 8,
                      "linear_value_head_dim": 8,
                      "linear_conv_kernel_dim": 2,
                      "vocab_size": 16,
                      "full_attention_interval": 1,
                      "mtp_num_hidden_layers": 1,
                      "tie_word_embeddings": true,
                      "rope_parameters": {
                        "type": "default",
                        "rope_theta": 100000.0,
                        "partial_rotary_factor": 0.25
                      }
                    }
                    """.utf8))
            let target = Qwen35TextModel(configuration)
            let drafter = Qwen35MTPDraftModel(configuration)
            for model in [target as Module, drafter as Module] {
                var weights = [String: MLXArray]()
                for (name, weight) in model.parameters().flattened() {
                    if name == "model.embed_tokens.weight" && !acceptDrafts {
                        weights[name] = broadcast(
                            (MLXArray(1 ... 16).asType(.float32) / 16)
                                .expandedDimensions(axis: -1),
                            to: weight.shape)
                    } else if name == "model.embed_tokens.weight"
                        || name.hasSuffix("norm.weight")
                    {
                        weights[name] = MLXArray.ones(weight.shape)
                    } else {
                        weights[name] = MLXArray.zeros(weight.shape)
                    }
                }
                try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
                model.apply { $0.asType(.float16) }
                eval(model)
            }

            let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
            let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
            var standard = try TokenIterator(input: input, model: target, parameters: parameters)
            var expected = [Int]()
            while let token = standard.next() {
                expected.append(token)
            }

            let cache = try target.newCache(parameters: parameters)
            var speculative = try MTPSpeculativeTokenIterator(
                input: input, mainModel: target, drafter: drafter, mainCache: cache,
                parameters: parameters, blockSize: 2)
            var actual = [Int]()
            while let token = speculative.next() {
                actual.append(token)
            }

            #expect(actual == expected)
            #expect(actual.count == 8)
            #expect(actual.allSatisfy { $0 == (acceptDrafts ? 0 : 15) })
            #expect(speculative.passthroughReason == nil)
            #expect(speculative.proposedCount > 0)
            if acceptDrafts {
                #expect(speculative.acceptedCount == speculative.proposedCount)
            } else {
                #expect(speculative.acceptedCount == 0)
            }
            let telemetry = try #require(speculative.speculativeDecodingTelemetry)
            #expect(telemetry.roundCount > 0)
            #expect(telemetry.targetModelCallCount > 0)
            #expect(telemetry.draftModelCallCount > 0)
            #expect(telemetry.emittedTokenCount == actual.count)
            #expect(cache.allSatisfy { $0.offset >= input.text.tokens.size })
        }
    }
}

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

extension MLXTestingSuite {
    @Suite
    struct ForkGenerationTests {
        @Test("Chunked generation preserves chosen token probabilities")
        func chosenProbabilitiesMatchModel() throws {
            try verifyChosenProbabilities(dtype: .float16)
        }

        @Test("Float32 CPU generation preserves chosen token probabilities")
        func chosenProbabilitiesMatchCPUModel() throws {
            try Device.withDefaultDevice(.cpu) {
                try verifyChosenProbabilities(dtype: .float32)
            }
        }

        private func verifyChosenProbabilities(dtype: DType) throws {
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

            let model = LlamaModel(
                LlamaConfiguration(
                    hiddenSize: 16, hiddenLayers: 1, intermediateSize: 32, attentionHeads: 2,
                    rmsNormEps: 0.00001, vocabularySize: 32, kvHeads: 1,
                    tieWordEmbeddings: false))
            var fixtureWeights: [String: MLXArray] = [:]
            for (name, weight) in model.parameters().flattened() {
                if name == "lm_head.weight" {
                    let rowValues = MLXArray(1 ... 32).asType(.float32) / 32
                    fixtureWeights[name] = broadcast(
                        rowValues.expandedDimensions(axis: -1), to: weight.shape)
                } else if name == "model.embed_tokens.weight" || name.hasSuffix("norm.weight") {
                    fixtureWeights[name] = MLXArray.ones(weight.shape)
                } else {
                    fixtureWeights[name] = MLXArray.zeros(weight.shape)
                }
            }
            try model.update(
                parameters: ModuleParameters.unflattened(fixtureWeights), verify: [.all])
            model.apply { $0.asType(dtype) }
            eval(model)
            let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7]
            let parameters = GenerateParameters(
                maxTokens: 2, temperature: 0,
                prefill: .init(stepSize: 3, chunking: .balanced))
            let cache = try model.newCache(parameters: parameters)
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(prompt)), model: model,
                cache: cache, parameters: parameters)
            #expect(cache.allSatisfy { $0.offset == prompt.count })

            var prefix = prompt
            for _ in 0 ..< 2 {
                let allLogits = model(MLXArray(prefix, [1, prefix.count]), cache: nil)
                let logits = allLogits[0..., -1, 0...]
                let expectedToken = argMax(logits, axis: -1).item(Int.self)
                let expectedLogProbability = logSoftmax(logits)[0, expectedToken].item(Float.self)
                let nextToken = iterator.next()
                let token = try #require(nextToken)
                let probabilities = try #require(iterator.lastLogProbabilities).materialize()

                #expect(token == expectedToken)
                #expect(probabilities.chosen.token == token)
                #expect(
                    abs(probabilities.chosen.logProbability - expectedLogProbability) < 0.0001,
                    "Expected \(expectedLogProbability), received \(probabilities.chosen.logProbability)"
                )
                #expect(abs(iterator.lastLogprob - expectedLogProbability) < 0.0001)
                #expect(probabilities.topLogProbabilities.isEmpty)
                #expect(iterator.streamingError == nil)
                prefix.append(Int32(token))
            }
            let exhausted = iterator.next()
            #expect(exhausted == nil)
            #expect(iterator.tokenCount == 2)
        }
    }
}

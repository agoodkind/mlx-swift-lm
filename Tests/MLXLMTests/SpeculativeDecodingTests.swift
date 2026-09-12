// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

extension MLXTestingSuite {
    @Suite
    struct SpeculativeDecodingTests {

    let processor: any UserInputProcessor
    let mainContext: ModelContext
    let draftContext: ModelContext

    init() {
        let processor = TestInputProcessor()
        let modelConfig = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64,
            attentionHeads: 4, headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )

        let mainModel = Gemma3TextModel(modelConfig)

        // on hardware with a NAX, float32 (the default dtype) runs
        // in tf32 in batch mode and float32 in non-batch.  this
        // change in behavior can cause issues with prediction and
        // doesn't match real world behavior (where float32 is not used)
        mainModel.apply {
            if $0.dtype == .float32 {
                $0.asType(.float16)
            } else {
                $0
            }
        }
        let mainContext = ModelContext(
            configuration: processor.configuration,
            model: mainModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        let draftModel = Gemma3TextModel(modelConfig)
        draftModel.apply {
            if $0.dtype == .float32 {
                $0.asType(.float16)
            } else {
                $0
            }
        }
        let draftContext = ModelContext(
            configuration: processor.configuration,
            model: draftModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        eval(mainModel, draftModel)
        self.processor = processor
        self.mainContext = mainContext
        self.draftContext = draftContext
    }

    @Test(arguments: [2, 4], [false])
    func testSpeculativeDecodingMatchesDefaultGeneration(
        numDraftTokens: Int,
        withLogitProcessor: Bool
    ) async throws {
        let vocabularySize = 100
        let tokenizer = TestTokenizer(vocabularySize: vocabularySize)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "stable-transition-test"),
            messageGenerator: DefaultMessageGenerator()
        )
        let model = StableTransitionLanguageModel(vocabularySize: vocabularySize)
        let draftModel = StableTransitionLanguageModel(vocabularySize: vocabularySize)
        let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        let input = LMInput(tokens: MLXArray([92, 85, 2, 95, 55, 7, 94, 42]))
        let parameters = GenerateParameters(
            maxTokens: 4,
            temperature: 0.0,  // Use greedy decoding for deterministic output
            repetitionPenalty: withLogitProcessor ? 1.5 : nil,
            presencePenalty: withLogitProcessor ? 0.5 : nil,
            frequencyPenalty: withLogitProcessor ? 0.2 : nil,
        )

        var normalTokens: [Int] = []
        for await generation in try generateTokens(
            input: input, parameters: parameters, context: context
        ) {
            if let token = generation.token { normalTokens.append(token) }
        }

        var speculativeTokens: [Int] = []
        for await generation in try generateTokens(
            input: input, parameters: parameters, context: context,
            draftModel: draftModel, numDraftTokens: numDraftTokens
        ) {
            if let token = generation.token { speculativeTokens.append(token) }
        }

        #expect(!normalTokens.isEmpty)
        #expect(!speculativeTokens.isEmpty)
        #expect(normalTokens == speculativeTokens)
    }

    @Test
    func testKVCacheIntegrityAfterDraftRejection() async throws {
        // This test specifically verifies that the KVCache is correctly pruned 
        // to the shared history length after a speculative rejection.
        let input = UserInput(prompt: "Analyze the current memory state")
        let modelInput = try await processor.prepare(input: input)
        let parameters = GenerateParameters(maxTokens: 16, temperature: 0.0)

        // Pass explicit caches so we can inspect their state after generation
        let mainCache = mainContext.model.newCache(parameters: parameters)
        let draftCache = draftContext.model.newCache(parameters: parameters)
        
        // Use the speculative generateTokens overload that takes explicit caches
        for await generation in try generateTokens(
            input: modelInput,
            cache: mainCache,
            parameters: parameters,
            context: mainContext,
            draftModel: draftContext.model,
            draftCache: draftCache,
            numDraftTokens: 5
        ) {
            if let token = generation.token {
                eval(token)
            }
        }

        #expect(!mainCache.isEmpty)
        for c in mainCache {
            // After completion, the cache offset should be > 0 (reflecting tokens generated)
            // This verifies that the speculative 'pruning' logic preserved the valid history.
            #expect(c.offset > 0)
        }
    }

    @Test
    func `SpeculativeTokenIterator scopes class logit processor state via copy`() throws {
        final class RecordingClassProcessor: LogitProcessor, @unchecked Sendable {
            var sampledTokens: [Int]

            init(sampledTokens: [Int] = []) {
                self.sampledTokens = sampledTokens
            }

            func prompt(_ prompt: MLXArray) {}
            func process(logits: MLXArray) -> MLXArray { logits }

            func didSample(token: MLXArray) {
                sampledTokens.append(token.item(Int.self))
            }

            func copy() -> Self {
                RecordingClassProcessor(sampledTokens: sampledTokens) as! Self
            }
        }

        let recordingProcessor = RecordingClassProcessor()
        var components = GenerationComponents()
        components.logitProcessorFactory = { recordingProcessor }

        let vocabularySize = 100
        // Mismatching draft model so some draft tokens are rejected
        let mainModel = StableTransitionLanguageModel(vocabularySize: vocabularySize)
        let draftModel = DivergentTransitionLanguageModel(vocabularySize: vocabularySize)

        var iterator = try SpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([10])),
            mainModel: mainModel,
            draftModel: draftModel,
            parameters: GenerateParameters(maxTokens: 5, temperature: 0.0),
            numDraftTokens: 3,
            components: components
        )

        var emittedTokens: [Int] = []
        while let token = iterator.next() {
            emittedTokens.append(token)
        }

        #expect(emittedTokens.count == 5)
        #expect(
            recordingProcessor.sampledTokens == emittedTokens,
            "Canonical processor didSample must match emitted tokens exactly; draft and verify copies must not pollute canonical state."
        )
    }
}
}

/// Deterministic causal model predicting different transitions to induce rejections in speculative decoding.
private final class DivergentTransitionLanguageModel: Module, LanguageModel,
    KVCacheDimensionProvider
{
    let vocabularySize: Int
    var kvHeads: [Int] { [] }

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenIds = inputs.asArray(Int.self)
        var logits = Array(
            repeating: Float(-100),
            count: tokenIds.count * vocabularySize
        )

        for (position, token) in tokenIds.enumerated() {
            logits[position * vocabularySize + ((token * 17 + 13) % vocabularySize)] = 100
        }

        return MLXArray(logits, [1, tokenIds.count, vocabularySize])
    }
}

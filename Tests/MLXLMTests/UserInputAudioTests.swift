// Copyright © 2026 Apple Inc.

import AVFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@Suite("UserInput.Audio source")
struct UserInputAudioTests {

    @Test("Encoded audio preserves URL decoding and mono samples")
    func encodedAudioMatchesURLDecoding() async throws {
        let url = try makeAudioFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let data = try Data(contentsOf: url)

        try await Device.withDefaultDevice(.cpu) {
            let encoded = UserInput.Audio.data(data, format: "wav")
            let reference = try MediaProcessing.extractAudioSamples(from: .url(url))
            let compatibilitySamples = try MediaProcessing.extractAudioSamples(from: encoded)
            var processing = UserInput.AudioProcessing()
            processing.sampleRate = 16_000
            let decoded = try await encoded.asMLXArray(processing: processing)
            let decodedSamples = decoded.asArray(Float.self)

            #expect(reference.count == 160)
            #expect(compatibilitySamples == reference)
            #expect(decodedSamples.count == reference.count)
            for (actual, expected) in zip(decodedSamples, reference) {
                #expect(abs(actual - expected) < 0.00001)
            }
            let arraySamples = try MediaProcessing.extractAudioSamples(from: .array(decoded))
            #expect(arraySamples == decodedSamples)
        }
    }

    @Test("Chat input preserves both public audio labels")
    func audioLabelsPreserveChatResources() {
        let source = UserInput.Audio.url(URL(fileURLWithPath: "/tmp/source.wav"))
        for input in [
            UserInput(prompt: "Listen", audio: [source]),
            UserInput(prompt: "Listen", audios: [source]),
            UserInput(prompt: .chat([.user("Listen", audios: [source])])),
        ] {
            #expect(input.audio.count == 1)
            #expect(input.audios.count == 1)
            guard case .chat(let messages) = input.prompt else {
                Issue.record("Audio input did not retain its chat prompt.")
                continue
            }
            #expect(messages.first?.audio.count == 1)
            #expect(messages.first?.audios.count == 1)
        }
    }

    private func makeAudioFile() throws -> URL {
        let format = try #require(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                channels: 1, interleaved: false))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 160))
        buffer.frameLength = 160
        let channels = try #require(buffer.floatChannelData)
        for index in 0 ..< 160 {
            channels[0][index] = Float(sin(Double(index) * 0.1)) * 0.25
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        return url
    }

    @Test("The url factory makes a url source")
    func urlFactoryMakesAURLSource() {
        let audio = UserInput.Audio.url(URL(fileURLWithPath: "/tmp/example.wav"))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/example.wav")
    }

    @Test("The array factory makes an array source")
    func arrayFactoryMakesAnArraySource() {
        let audio = UserInput.Audio.array(MLXArray([0.25, -0.5, 1] as [Float]))
        guard case .array(let array) = audio.source else {
            Issue.record("expected an array source")
            return
        }
        #expect(array.asArray(Float.self) == [0.25, -0.5, 1])
    }

    @Test("A factory works as an implicit member")
    func factoryWorksAsAnImplicitMember() {
        let audio: UserInput.Audio = .url(URL(fileURLWithPath: "/tmp/implicit.wav"))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/implicit.wav")
    }

    @Test("A factory can be passed where a function of one argument is expected")
    func factoryWorksAsAFunctionValue() {
        let urls = [URL(fileURLWithPath: "/tmp/a.wav"), URL(fileURLWithPath: "/tmp/b.wav")]
        let audios = urls.map(UserInput.Audio.url)
        let paths = audios.compactMap { audio -> String? in
            guard case .url(let url) = audio.source else { return nil }
            return url.path
        }
        #expect(paths == ["/tmp/a.wav", "/tmp/b.wav"])
    }

    @Test("A factory works as a function value without a type context")
    func factoryWorksWithoutATypeContext() {
        let make = UserInput.Audio.url
        let audio = make(URL(fileURLWithPath: "/tmp/bare.wav"))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/bare.wav")
    }

    @Test("The initializer stores the source")
    func initializerStoresTheSource() {
        let audio = UserInput.Audio(source: .url(URL(fileURLWithPath: "/tmp/init.wav")))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/init.wav")
    }
}

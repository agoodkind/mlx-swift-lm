// Copyright © 2025 Apple Inc.

import MLX

private let messageContentLogger = MLXLogger(label: "MessageContent")

package enum MessageContentLayout: Sendable {
    case imagesThenVideosThenText
    case imagesThenText
    case textThenImages
}

public enum Chat {
    public struct Message {
        /// The role of the message sender.
        public var role: Role

        /// The content of the message.
        public var content: String

        /// Array of image data associated with the message.
        public var images: [UserInput.Image]

        /// Array of video data associated with the message.
        public var videos: [UserInput.Video]

        /// Array of audio data associated with the message.
        public var audio: [UserInput.Audio]

        /// Array of tool calls (typically for the assistant role).
        public var toolCalls: [[String: any Sendable]]?

        /// Optional ID of the tool call this message responds to (typically for the tool role).
        public var toolCallId: String?

        public init(
            role: Role, content: String, images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [], audio: [UserInput.Audio] = [], toolCalls: [[String: any Sendable]]? = nil,
            toolCallId: String? = nil
        ) {
            self.role = role
            self.content = content
            self.images = images
            self.videos = videos
            self.audio = audio
            self.toolCalls = toolCalls
            self.toolCallId = toolCallId
        }

        public static func system(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = [], audio: [UserInput.Audio] = []
        ) -> Self {
            Self(role: .system, content: content, images: images, videos: videos, audio: audio)
        }

        public static func assistant(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = [], audio: [UserInput.Audio] = [],
            toolCalls: [[String: any Sendable]]? = nil
        ) -> Self {
            Self(role: .assistant, content: content, images: images, videos: videos, audio: audio, toolCalls: toolCalls)
        }

        public static func user(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = [], audio: [UserInput.Audio] = []
        ) -> Self {
            Self(role: .user, content: content, images: images, videos: videos, audio: audio)
        }

        public static func tool(_ content: String, toolCallId: String? = nil) -> Self {
            Self(role: .tool, content: content, toolCallId: toolCallId)
        }

        public enum Role: String, Sendable {
            case user
            case assistant
            case system
            case tool
        }
    }
}

/// Protocol for something that can convert structured
/// ``Chat/Message`` into model specific ``Message``
/// (raw dictionary) format.
///
/// Typically this is owned and used by a ``UserInputProcessor``:
///
/// ```swift
/// public func prepare(input: UserInput) async throws -> LMInput {
///     let messages = Qwen2VLMessageGenerator().generate(from: input)
///     ...
/// ```
public protocol MessageGenerator: Sendable {

    /// Generates messages from the input.
    func generate(from input: UserInput) -> [Message]

    /// Returns array of `[String: any Sendable]` aka ``Message``
    func generate(messages: [Chat.Message]) -> [Message]

    /// Returns `[String: any Sendable]`, aka ``Message``.
    func generate(message: Chat.Message) -> Message
}

extension MessageGenerator {

    public func generate(message: Chat.Message) -> Message {
        var dict: [String: any Sendable] = [
            "role": message.role.rawValue,
            "content": message.content,
        ]
        if let toolCalls = message.toolCalls {
            dict["tool_calls"] = toolCalls
        }
        if let toolCallId = message.toolCallId {
            dict["tool_call_id"] = toolCallId
        }
        return dict
    }

    package func contentParts(
        for message: Chat.Message, layout: MessageContentLayout
    ) -> [[String: String]] {
        var parts: [[String: String]] = []

        func appendText(_ text: String) {
            parts.append(["type": "text", "text": text])
        }

        func appendImages() {
            for image in message.images {
                if let marker = image.labelMarkerCharacter {
                    messageContentLogger.warning(
                        "Leaving an image name out of the prompt: it holds `\(marker)`, which vision models build their image placeholders from"
                    )
                } else if let label = image.label {
                    appendText(UserInput.Image.promptText(forLabel: label))
                }
                parts.append(["type": "image"])
            }
        }

        switch layout {
        case .imagesThenVideosThenText:
            appendImages()
            parts.append(contentsOf: message.videos.map { _ in ["type": "video"] })
            appendText(message.content)
        case .imagesThenText:
            appendImages()
            appendText(message.content)
        case .textThenImages:
            appendText(message.content)
            appendImages()
        }

        return parts
    }

    public func generate(messages: [Chat.Message]) -> [Message] {
        var rawMessages: [Message] = []

        for message in messages {
            let raw = generate(message: message)
            rawMessages.append(raw)
        }

        return rawMessages
    }

    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            generate(messages: [.user(text)])
        case .messages(let messages):
            messages
        case .chat(let messages):
            generate(messages: messages)
        }
    }
}

/// Default implementation of ``MessageGenerator`` that produces `role` and
/// `content`, plus `name`, `tool_call_id`, and `tool_calls` when present.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct DefaultMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        var dict: [String: any Sendable] = [
            "role": message.role.rawValue,
            "content": message.content,
        ]
        if let toolCalls = message.toolCalls {
            dict["tool_calls"] = toolCalls
        }
        if let toolCallId = message.toolCallId {
            dict["tool_call_id"] = toolCallId
        }
        return dict
    }
}

/// Implementation of ``MessageGenerator`` that produces a
/// `role` and `content` but omits `system` roles.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct NoSystemMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(messages: [Chat.Message]) -> [Message] {
        messages
            .filter { $0.role != .system }
            .map { generate(message: $0) }
    }
}

extension UserInput {

    /// Returns the input unchanged if its prompt is not `.chat`. Call this method before a
    /// message generator turns the prompt into `.messages`.
    package func removingSpecialTokenLabels(using tokenizer: any Tokenizer) -> UserInput {
        guard case .chat(let messages) = prompt else { return self }
        var screened = self
        screened.prompt = .chat(
            messages.map { message in
                var message = message
                message.images = message.images.map { image in
                    guard let label = image.label,
                        let names = tokenizer.specialTokenNames(inImageLabel: label)
                    else { return image }
                    let named = names.isEmpty ? "a special token" : names.joined(separator: ", ")
                    messageContentLogger.warning(
                        "Leaving an image name out of the prompt, because this model's tokenizer reads it as \(named)"
                    )
                    var image = image
                    image.label = nil
                    return image
                }
                return message
            })
        return screened
    }
}

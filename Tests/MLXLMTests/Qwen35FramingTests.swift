import MLXLMCommon
import Testing

extension MLXTestingSuite {
    @Suite
    struct Qwen35FramingTests {
        static let payloads = [
            #"{"name":"lookup","arguments":{"city":"Paris"}}"#,
            "<function=lookup><parameter=city>Paris</parameter></function>",
        ]

        static let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "lookup",
                    "parameters": ["type": "object"],
                ] as [String: any Sendable],
            ]
        ]

        @Test(arguments: payloads)
        func rejectsPayloadWithoutOpeningFrame(_ payload: String) {
            let parser = Qwen35ToolCallParser(
                startTag: "<tool_call>", endTag: "</tool_call>")
            #expect(parser.parse(content: payload, tools: Self.tools) == nil)
            #expect(parser.parseEOS(payload, tools: Self.tools).isEmpty)

            let processor = ToolCallProcessor(
                format: .qwen35, tools: Self.tools,
                toolCallPolicy: .init(recovery: .disabled))
            _ = processor.processChunk(payload)
            processor.processEOS()
            #expect(processor.toolCalls.isEmpty)
        }

        @Test(arguments: payloads)
        func acceptsOpeningFrameWithOptionalClosingFrame(_ payload: String) throws {
            for closingFrame in ["</tool_call>", ""] {
                let processor = ToolCallProcessor(
                    format: .qwen35, tools: Self.tools,
                    toolCallPolicy: .init(recovery: .disabled))
                _ = processor.processChunk("<tool_call>\(payload)\(closingFrame)")
                processor.processEOS()

                #expect(processor.toolCalls.count == 1)
                let call = try #require(processor.toolCalls.first)
                #expect(call.function.name == "lookup")
                #expect(call.function.arguments["city"] == .string("Paris"))
            }
        }

        @Test func parsesMultipleFramedCallsAtEndOfStream() {
            let parser = Qwen35ToolCallParser(
                startTag: "<tool_call>", endTag: "</tool_call>")
            let content = Self.payloads.map { "<tool_call>\($0)</tool_call>" }.joined()
            let calls = parser.parseEOS(content, tools: Self.tools)
            #expect(calls.count == 2)
            #expect(calls.allSatisfy { $0.function.arguments["city"] == .string("Paris") })
        }
    }
}

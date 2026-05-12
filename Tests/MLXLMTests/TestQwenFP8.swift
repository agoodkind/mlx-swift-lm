import XCTest

/// Placeholder for Qwen FP8 SSD-streaming integration test.
/// Requires the Qwen/Qwen3.6-35B-A3B-FP8 model to be cached locally.
/// Run manually via `swift test --filter TestQwenFP8` with the model present.
final class TestQwenFP8: XCTestCase {
    func testGeneration() throws {
        // Skipped: requires real model weights. Run manually when model is available.
        throw XCTSkip("Requires Qwen3.6-35B-A3B-FP8 weights locally — run manually.")
    }
}

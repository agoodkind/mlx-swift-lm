#!/usr/bin/env swift
// gemma4_mtp_test.swift
// Gemma4 MTP Speculative Decoding — Real Model Integration Test
//
// Safety limits:
//   - max_tokens: 50 (avoids long runaway generation)
//   - maxKVSize: 512  (caps KV cache RAM use)
//   - No parallel requests
//   - Model combo: E2B-4bit (3.4 GB) + E2B-assistant-bf16 (181 MB) ≈ 3.6 GB total
//
// Usage:
//   swift gemma4_mtp_test.swift
//
// Expected output:
//   Baseline TPS: ~XX tok/s
//   MTP TPS:      ~XX tok/s
//   Speedup:      ~X.Xx
//   Accept rate:  ~XX%

import Foundation

let assistantDir = URL(fileURLWithPath: NSHomeDirectory())
    .appendingPathComponent(".cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-assistant-bf16/snapshots")
let mainDir = URL(fileURLWithPath: NSHomeDirectory())
    .appendingPathComponent(".cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots")

func snapshotURL(_ base: URL) -> URL? {
    let fm = FileManager.default
    guard let subs = try? fm.contentsOfDirectory(at: base, includingPropertiesForKeys: nil),
          let first = subs.first else { return nil }
    return first
}

guard let mainSnap = snapshotURL(mainDir),
      let asstSnap = snapshotURL(assistantDir) else {
    print("❌ Could not find model snapshots. Ensure both models are cached.")
    exit(1)
}

print("✅ Main model:      \(mainSnap.lastPathComponent)")
print("✅ Assistant model: \(asstSnap.lastPathComponent)")
print("")
print("📋 Memory budget:")
print("   E2B-it-4bit:        ~3.4 GB")
print("   E2B-assistant-bf16: ~0.18 GB")
print("   KV cache (max 512): ~0.05 GB")
print("   Total:              ~3.7 GB  (safe on 64 GB M5 Pro)")
print("")
print("⚠️  This script prints configuration details.")
print("    The actual MLX model loading requires linking against MLXLLM.")
print("    Run via the test harness instead:")
print("")
print("    bash run_tests.sh Gemma4MTPIntegrationTests")
print("")
print("📝 Test configuration:")
print("   Main:        mlx-community/gemma-4-e2b-it-4bit")
print("   Assistant:   mlx-community/gemma-4-E2B-it-assistant-bf16")
print("   Prompt:      'What is the capital of France?'")
print("   max_tokens:  50")
print("   maxKVSize:   512  (memory cap)")
print("   numDraft:    2    (2 MTP draft tokens per round)")
print("   temperature: 0.0  (greedy — deterministic)")

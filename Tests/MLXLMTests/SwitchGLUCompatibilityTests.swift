import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

extension MLXTestingSuite {
    @Suite
    struct SwitchGLUCompatibilityTests {
        @Test(arguments: [2, 32])
        func rotatedExpertsApplyBothTransforms(tokenCount: Int) throws {
            let previousMemoryLimit = Memory.memoryLimit
            let previousCacheLimit = Memory.cacheLimit
            Memory.memoryLimit = 64 * 1024 * 1024
            Memory.cacheLimit = 0
            defer {
                Memory.clearCache()
                Memory.cacheLimit = previousCacheLimit
                Memory.memoryLimit = previousMemoryLimit
            }

            let layer = RotateSwitchGLU(
                inputDims: 16, hiddenDims: 8, numExperts: 4, groupSize: 8, krot: 1)
            for projection in [layer.gateProj, layer.upProj, layer.downProj] {
                let weight =
                    sin(
                        MLXArray(0 ..< projection.weight.size).asType(.float32) / 7
                    ).reshaped(projection.weight.shape).asType(.float16) * 0.1
                try projection.update(
                    parameters: ModuleParameters.unflattened(["weight": weight]), verify: [.all])
            }
            for (rotation, dimensions) in [(layer.gateUpRot, 16), (layer.downRot, 8)] {
                let pairs = (0 ..< dimensions).map { Int16($0 % 8) }
                try rotation.update(
                    parameters: ModuleParameters.unflattened([
                        "theta": MLXArray.ones([1, dimensions / 2]) * 0.25,
                        "pairs": MLXArray(pairs).reshaped(1, dimensions),
                        "channel_scales": MLXArray.ones([1, dimensions]) * 1.1,
                    ]), verify: [.all])
            }
            layer.apply { $0.dtype.isFloatingPoint ? $0.asType(.float16) : $0 }
            for rotation in [layer.gateUpRot, layer.downRot] {
                eval(rotation.prepareDerivedRotationState())
            }
            eval(layer)

            let input =
                MLXArray(0 ..< tokenCount * 16).asType(.float32)
                .reshaped(1, tokenCount, 16).asType(.float16) / Float(tokenCount * 16)
            let indices = MLXArray((0 ..< tokenCount * 2).map { UInt32(($0 * 3) % 4) })
                .reshaped(1, tokenCount, 2)
            let expanded = expandedDimensions(input, axes: [-2, -3])
            let rotatedInput = layer.gateUpRot.rotate(expanded)
            let gate = layer.gateProj(rotatedInput, indices)
            let up = layer.upProj(rotatedInput, indices)
            let hidden = silu(gate) * up
            let expected = layer.downProj(layer.downRot.rotate(hidden), indices)
                .squeezed(axis: -2)
            let withoutTransforms = layer.downProj(
                silu(layer.gateProj(expanded, indices)) * layer.upProj(expanded, indices),
                indices
            ).squeezed(axis: -2)
            let output = layer(input, indices)
            eval(output, expected, withoutTransforms)

            #expect(output.shape == [1, tokenCount, 2, 16])
            #expect(
                allClose(output, expected, rtol: 1e-4, atol: 1e-5).item(Bool.self),
                "Maximum absolute difference: \(abs(output - expected).max().item(Float.self))"
            )
            #expect(!allClose(output, withoutTransforms, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        }
    }
}

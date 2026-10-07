import AVFoundation
import CoreML
import XCTest

@testable import FluidAudio

final class CtcAudioPaddingTests: XCTestCase {
    func testRealRecordingCopyInitializesPaddingAcrossRanksAndDataTypes() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/01-validation-request-21.4s", withExtension: "wav")
                ?? Bundle.module.url(forResource: "01-validation-request-21.4s", withExtension: "wav"))
        let file = try AVAudioFile(forReading: url)
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let pointer = try XCTUnwrap(buffer.floatChannelData?[0])
        let recording = Array(UnsafeBufferPointer(start: pointer, count: Int(buffer.frameLength)))
        let start = try XCTUnwrap(recording.firstIndex(where: { abs($0) > 0.005 }))
        let samples = Array(recording[start..<min(start + 1024, recording.count)])
        XCTAssertEqual(samples.count, 1024)
        XCTAssertTrue(samples[128..<512].contains(where: { $0 != 0 }))

        for shape: [NSNumber] in [[512], [1, 512]] {
            for type: MLMultiArrayDataType in [.float32, .float16] {
                let array = try MLMultiArray(shape: shape, dataType: type)
                for inputCount in [0, 128, 1024] {
                    // Reused storage starts with real recorded speech, so an
                    // unwritten tail fails deterministically without allocator luck.
                    for index in 0..<array.count {
                        array[index] = NSNumber(value: samples[index])
                    }
                    let input = Array(samples.prefix(inputCount))
                    let copied = CtcKeywordSpotter.copyAndPadAudio(input, into: array)
                    XCTAssertEqual(copied, min(inputCount, array.count))
                    for index in 0..<copied {
                        XCTAssertEqual(array[index].floatValue, input[index], accuracy: 0.001)
                    }
                    XCTAssertTrue(
                        (copied..<array.count).allSatisfy { array[$0].floatValue == 0 },
                        "Every padded sample must be initialized for shape \(shape), type \(type)")
                }
            }
        }
    }
}

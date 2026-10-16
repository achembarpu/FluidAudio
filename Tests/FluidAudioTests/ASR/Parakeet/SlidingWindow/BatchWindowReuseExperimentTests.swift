#if DEBUG
import AVFoundation
import XCTest

import FluidAudio

/// Protects full-recording authority and recording isolation when prior batch windows are reused.
@MainActor
final class BatchWindowReuseExperimentTests: XCTestCase {
    private func assertTimingsEqual(
        _ actual: [TokenTiming]?, _ expected: [TokenTiming]?, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual?.map(\.token), expected?.map(\.token), file: file, line: line)
        XCTAssertEqual(actual?.map(\.tokenId), expected?.map(\.tokenId), file: file, line: line)
        XCTAssertEqual(actual?.map(\.startTime), expected?.map(\.startTime), file: file, line: line)
        XCTAssertEqual(actual?.map(\.endTime), expected?.map(\.endTime), file: file, line: line)
        XCTAssertEqual(actual?.map(\.confidence), expected?.map(\.confidence), file: file, line: line)
    }

    func testGrowingRealRecordingPreservesFinalTokensAndResetRejectsPreviousRecording() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_EXACT_WINDOW_EXPERIMENT"] == "1")
        #if SWIFT_PACKAGE
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/01-validation-request-21.4s", withExtension: "wav")
                ?? Bundle.module.url(forResource: "01-validation-request-21.4s", withExtension: "wav"))
        #else
        let url = URL(fileURLWithPath: try XCTUnwrap(ProcessInfo.processInfo.environment["NON_EOU_AUDIO"]))
        #endif
        let file = try AVAudioFile(forReading: url)
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let channel = try XCTUnwrap(buffer.floatChannelData?[0])
        let samples = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
        XCTAssertEqual(file.processingFormat.sampleRate, 16_000)
        for version: AsrModelVersion in [.v2, .v3, .tdtCtc110m] {
            let models = try AsrModels.loadLocal(from: AsrModels.defaultCacheDirectory(for: version), version: version)
            let manager = AsrManager(models: models)
            func decode(_ audio: [Float]) async throws -> ASRResult {
                var state = TdtDecoderState.make(decoderLayers: version.decoderLayers)
                return try await manager.transcribe(audio, decoderState: &state, language: .english)
            }
            let baseline = try await decode(samples)
            await manager.beginBatchWindowReuseExperiment()
            _ = try await decode(Array(samples.prefix(16 * 16_000)))
            _ = try await decode(Array(samples.prefix(19 * 16_000)))
            let cached = try await decode(samples)
            XCTAssertEqual(cached.text, baseline.text)
            XCTAssertEqual(cached.confidence, baseline.confidence)
            XCTAssertEqual(cached.duration, baseline.duration)
            assertTimingsEqual(cached.tokenTimings, baseline.tokenTimings)
            let reused = await manager.batchWindowReuseExperimentStatistics()
            XCTAssertGreaterThan(reused.hits, 0)

            await manager.reset()
            let reset = await manager.batchWindowReuseExperimentStatistics()
            XCTAssertEqual(reset.entries, 0)
            let nextRecording = try await decode(Array(samples.dropFirst(16_000)))
            let independent = AsrManager(models: models)
            var state = TdtDecoderState.make(decoderLayers: version.decoderLayers)
            let nextBaseline = try await independent.transcribe(
                Array(samples.dropFirst(16_000)), decoderState: &state, language: .english)
            XCTAssertEqual(nextRecording.text, nextBaseline.text)
            assertTimingsEqual(nextRecording.tokenTimings, nextBaseline.tokenTimings)
            await manager.cleanup()
            await independent.cleanup()
        }
    }
}
#endif

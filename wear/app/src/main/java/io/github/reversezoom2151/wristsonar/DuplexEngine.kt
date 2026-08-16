package io.github.reversezoom2151.wristsonar

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.SystemClock
import java.util.concurrent.atomic.AtomicBoolean

data class AudioHealth(
    val actualSampleRate: Int,
    val agcEnabled: Boolean?,
    val noiseSuppressorEnabled: Boolean?,
    val outputUnderruns: Int,
)

/**
 * Performs true duplex capture and emits unaligned raw microphone callbacks.
 *
 * Alignment is intentionally left to the host `PcmSynchronizer`. Android's
 * callback boundaries cannot be treated as chirp boundaries just because the
 * app writes one chirp per loop.
 */
class DuplexEngine(
    private val onFrame: (ShortArray, Long, AudioHealth) -> Unit,
    private val onFailure: (Throwable) -> Unit,
) {
    private val running = AtomicBoolean(false)
    private var worker: Thread? = null
    private var record: AudioRecord? = null
    private var track: AudioTrack? = null
    private var agc: AutomaticGainControl? = null
    private var suppressor: NoiseSuppressor? = null

    fun start() {
        check(running.compareAndSet(false, true)) { "duplex engine is already running" }
        worker = Thread(::run, "wristsonar-duplex").also { it.start() }
    }

    fun stop() {
        running.set(false)
        worker?.join(1_000)
        worker = null
        release()
    }

    private fun run() {
        try {
            val chirp = Fmcw.pcm16()
            val input = createRecord()
            val output = createTrack()
            record = input
            track = output
            check(input.sampleRate == Fmcw.sampleRate) {
                "device delivered ${input.sampleRate} Hz, required ${Fmcw.sampleRate} Hz"
            }
            agc = AutomaticGainControl.create(input.audioSessionId)?.also { it.enabled = false }
            suppressor = NoiseSuppressor.create(input.audioSessionId)?.also { it.enabled = false }
            input.startRecording()
            output.play()
            while (running.get()) {
                val written = output.write(chirp, 0, chirp.size, AudioTrack.WRITE_BLOCKING)
                check(written == chirp.size) { "AudioTrack wrote $written of ${chirp.size} samples" }
                val captured = ShortArray(Fmcw.frameSamples)
                val read = input.read(captured, 0, captured.size, AudioRecord.READ_BLOCKING)
                check(read == captured.size) { "AudioRecord read $read of ${captured.size} samples" }
                val health = AudioHealth(
                    actualSampleRate = input.sampleRate,
                    agcEnabled = agc?.enabled,
                    noiseSuppressorEnabled = suppressor?.enabled,
                    outputUnderruns = output.underrunCount,
                )
                onFrame(captured, SystemClock.elapsedRealtimeNanos(), health)
            }
        } catch (failure: Throwable) {
            if (running.get()) onFailure(failure)
        } finally {
            release()
            running.set(false)
        }
    }

    private fun createRecord(): AudioRecord {
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(Fmcw.sampleRate)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .build()
        val minimum = AudioRecord.getMinBufferSize(
            Fmcw.sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        return AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
            .setAudioFormat(format)
            .setBufferSizeInBytes(maxOf(minimum, Fmcw.frameSamples * 2 * 4))
            .build()
    }

    private fun createTrack(): AudioTrack {
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(Fmcw.sampleRate)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        return AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build(),
            )
            .setAudioFormat(format)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(Fmcw.frameSamples * 2 * 4)
            .build()
    }

    private fun release() {
        runCatching { record?.stop() }
        runCatching { track?.pause() }
        agc?.release()
        suppressor?.release()
        record?.release()
        track?.release()
        agc = null
        suppressor = null
        record = null
        track = null
    }
}

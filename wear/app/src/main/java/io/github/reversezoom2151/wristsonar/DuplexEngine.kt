package io.github.reversezoom2151.wristsonar

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTimestamp
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean

/**
 * What the capture path actually did, as opposed to what it was asked to do.
 *
 * @param nominalSampleRate the rate the stream is declared to be on. This is
 *   the rate that was requested: `AudioRecord.getSampleRate()` returns the
 *   requested rate, so comparing it to the constant proves nothing at all.
 * @param effectiveSampleRate frames actually delivered per second of boot
 *   time, recovered by differencing `AudioRecord.getTimestamp()` frame
 *   positions. Null until the measurement baseline is long enough, or on a
 *   device whose HAL declines to report a timestamp. This is the only field
 *   that can catch the platform silently resampling.
 * @param unprocessedSource whether the recorder opened `UNPROCESSED`, the one
 *   public source defined to carry no gain control or noise suppression.
 * @param agcEnabled null when no `AutomaticGainControl` handle exists. Null is
 *   not good news: it is the common case where gain control lives in the HAL,
 *   where it cannot be observed or switched off. AGC across 18 to 21 kHz
 *   destroys exactly the amplitude relationships the differential profile is
 *   made of.
 * @param newOutputUnderruns underruns since the previous frame. The platform
 *   counter is cumulative and never resets, so only the delta says whether
 *   anything is wrong now.
 */
data class AudioHealth(
    val nominalSampleRate: Int,
    val effectiveSampleRate: Double?,
    val unprocessedSource: Boolean,
    val agcEnabled: Boolean?,
    val noiseSuppressorEnabled: Boolean?,
    val newOutputUnderruns: Int,
    val totalOutputUnderruns: Int,
)

/**
 * Performs true duplex capture and emits unaligned raw microphone callbacks.
 *
 * Alignment is intentionally left to the host `PcmSynchronizer`. Android's
 * callback boundaries cannot be treated as chirp boundaries just because the
 * app writes one chirp per loop.
 *
 * `onFrame` runs on the capture thread and must not block on anything slower
 * than the frame period. Anything that can block, TCP in particular, belongs
 * behind a queue in the caller.
 */
class DuplexEngine(
    private val unprocessedSupported: Boolean,
    private val onFrame: (ShortArray, Long, AudioHealth) -> Unit,
    private val onFailure: (Throwable) -> Unit,
) {
    private val running = AtomicBoolean(false)

    @Volatile
    private var worker: Thread? = null

    @Volatile
    private var record: AudioRecord? = null

    @Volatile
    private var track: AudioTrack? = null
    private var agc: AutomaticGainControl? = null
    private var suppressor: NoiseSuppressor? = null
    private var unprocessed = false

    fun start() {
        check(running.compareAndSet(false, true)) { "duplex engine is already running" }
        worker = Thread(::run, "wristsonar-duplex").also { it.start() }
    }

    /**
     * Stop the loop and free the microphone, in that order.
     *
     * The previous version released the devices while the worker could still
     * be parked inside `read` or `write`, and it never guaranteed the
     * microphone was handed back at all if the worker was wedged. Stopping the
     * recorder unblocks a parked read, and nothing is released until the
     * worker has actually finished.
     */
    fun stop() {
        if (!running.compareAndSet(true, false)) {
            release()
            return
        }
        runCatching { record?.stop() }
            .onFailure { Log.w(TAG, "could not stop the recorder", it) }
        val thread = worker
        worker = null
        thread?.join(STOP_JOIN_MILLIS)
        if (thread != null && thread.isAlive) {
            // Releasing now would free objects the worker is still inside.
            Log.e(TAG, "duplex worker still running after $STOP_JOIN_MILLIS ms")
            return
        }
        release()
    }

    private fun run() {
        try {
            val chirp = Fmcw.pcm16()
            val input = createRecord()
            record = input
            val output = createTrack()
            track = output
            check(input.state == AudioRecord.STATE_INITIALIZED) {
                "AudioRecord did not initialise for ${Fmcw.sampleRate} Hz mono PCM"
            }
            check(output.state == AudioTrack.STATE_INITIALIZED) {
                "AudioTrack did not initialise for ${Fmcw.sampleRate} Hz mono PCM"
            }
            agc = AutomaticGainControl.create(input.audioSessionId)?.also {
                it.enabled = false
            }
            suppressor = NoiseSuppressor.create(input.audioSessionId)?.also {
                it.enabled = false
            }

            // Fill the output ring before starting it. Calling play() on an
            // empty track guarantees a startup underrun, and an underrun is a
            // shift in the transmit cadence the host has to re-lock to.
            repeat(PREROLL_FRAMES) {
                val primed = output.write(chirp, 0, chirp.size, AudioTrack.WRITE_BLOCKING)
                check(primed == chirp.size) { "AudioTrack pre-roll wrote $primed" }
            }
            input.startRecording()
            output.play()

            val clock = CaptureClock()
            var framesRead = 0L
            var previousUnderruns = output.underrunCount
            while (running.get()) {
                val captured = ShortArray(Fmcw.frameSamples)
                val read = input.read(captured, 0, captured.size, AudioRecord.READ_BLOCKING)
                check(read == captured.size) {
                    "AudioRecord read $read of ${captured.size} samples"
                }
                clock.update(input)
                // The wire format defines a packet stamp as the capture time
                // of the first sample. The fallback back-dates a stamp taken
                // after read() returned; the timestamp path is better because
                // it does not inherit this thread's scheduling delay at all.
                val fallback = SystemClock.elapsedRealtimeNanos() - Fmcw.frameDurationNanos
                val startNanos = clock.startNanos(framesRead, fallback)
                framesRead += read.toLong()

                val underruns = output.underrunCount
                val health = AudioHealth(
                    nominalSampleRate = Fmcw.sampleRate,
                    effectiveSampleRate = clock.effectiveSampleRate(RATE_BASELINE_SECONDS),
                    unprocessedSource = unprocessed,
                    agcEnabled = agc?.enabled,
                    noiseSuppressorEnabled = suppressor?.enabled,
                    newOutputUnderruns = (underruns - previousUnderruns).coerceAtLeast(0),
                    totalOutputUnderruns = underruns,
                )
                previousUnderruns = underruns
                onFrame(captured, startNanos, health)

                val written = output.write(chirp, 0, chirp.size, AudioTrack.WRITE_BLOCKING)
                check(written == chirp.size) {
                    "AudioTrack wrote $written of ${chirp.size} samples"
                }
            }
        } catch (failure: Throwable) {
            // Never swallowed. First bring-up on real hardware is undiagnosable
            // if the only symptom is a service that quietly went away.
            Log.e(TAG, "duplex loop ended", failure)
            if (running.get()) onFailure(failure)
        } finally {
            release()
            running.set(false)
        }
    }

    private fun createRecord(): AudioRecord {
        // UNPROCESSED is the only public source defined to skip gain control
        // and noise suppression. VOICE_RECOGNITION is merely the least
        // speech-processed of the rest, which is not the same claim, so it is
        // used only where the device says UNPROCESSED is unavailable and the
        // caller is told which one it got.
        unprocessed = unprocessedSupported
        val source = if (unprocessedSupported) {
            MediaRecorder.AudioSource.UNPROCESSED
        } else {
            Log.w(TAG, "device does not support UNPROCESSED; falling back")
            MediaRecorder.AudioSource.VOICE_RECOGNITION
        }
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
        // About 100 ms of headroom. The old 50 ms was sized for a loop that
        // wrote to a socket inline, where one slow send overflowed the ring
        // and silently deleted samples the host counts on being contiguous.
        val wanted = Fmcw.frameSamples * 2 * 8
        return AudioRecord.Builder()
            .setAudioSource(source)
            .setAudioFormat(format)
            .setBufferSizeInBytes(maxOf(minimum, wanted))
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

    @Synchronized
    private fun release() {
        runCatching { record?.stop() }
        runCatching { track?.pause() }
        runCatching { track?.flush() }
        agc?.release()
        suppressor?.release()
        record?.release()
        track?.release()
        agc = null
        suppressor = null
        record = null
        track = null
    }

    /**
     * Maps a microphone frame index onto the boot-time clock.
     *
     * `AudioRecord.getTimestamp` reports which frame was at the microphone at
     * a given instant, which is a hardware fact rather than a statement about
     * when this thread happened to run. Differencing two of them over seconds
     * also recovers the rate the device is truly delivering, which is the only
     * way to notice the platform resampling underneath a 48 kHz request.
     */
    private class CaptureClock {
        private val probe = AudioTimestamp()
        private var anchorFrames = -1L
        private var anchorNanos = 0L
        private var latestFrames = -1L
        private var latestNanos = 0L

        fun update(record: AudioRecord) {
            val status = record.getTimestamp(probe, AudioTimestamp.TIMEBASE_BOOTTIME)
            if (status != AudioRecord.SUCCESS) return
            if (probe.framePosition <= latestFrames) return
            if (anchorFrames < 0L) {
                anchorFrames = probe.framePosition
                anchorNanos = probe.nanoTime
            }
            latestFrames = probe.framePosition
            latestNanos = probe.nanoTime
        }

        /** Frames per second of boot time, or null until the baseline is long. */
        fun effectiveSampleRate(minimumSeconds: Double): Double? {
            if (anchorFrames < 0L || latestFrames <= anchorFrames) return null
            val seconds = (latestNanos - anchorNanos) / 1_000_000_000.0
            if (seconds < minimumSeconds) return null
            return (latestFrames - anchorFrames) / seconds
        }

        /** Boot-time nanoseconds at which `frameIndex` reached the microphone. */
        fun startNanos(frameIndex: Long, fallbackNanos: Long): Long {
            if (latestFrames < 0L) return maxOf(fallbackNanos, 0L)
            val offset = (frameIndex - latestFrames).toDouble() * 1_000_000_000.0
            val estimate = latestNanos + (offset / Fmcw.sampleRate).toLong()
            return maxOf(estimate, 0L)
        }
    }

    private companion object {
        const val TAG = "WristsonarDuplex"
        const val PREROLL_FRAMES = 2
        const val STOP_JOIN_MILLIS = 2_000L
        const val RATE_BASELINE_SECONDS = 2.0
    }
}

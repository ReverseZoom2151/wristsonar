package io.github.reversezoom2151.wristsonar

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/** The transmitted waveform must stay bit-for-bit compatible with the host. */
object Fmcw {
    const val sampleRate = 48_000
    const val frameSamples = 600
    const val startHz = 18_000.0
    const val endHz = 21_000.0
    private const val tukeyAlpha = 0.25
    private const val amplitude = 0.20

    /**
     * Nanoseconds of audio in one transmitted frame, 12.5 ms.
     *
     * Used to back-date a stamp taken once a read has returned, because the
     * host wire format defines a packet timestamp as the capture time of the
     * first sample and not the last. Reading it the other way is a fixed sign
     * error of exactly this constant.
     */
    val frameDurationNanos: Long = frameSamples.toLong() * 1_000_000_000L / sampleRate

    /** A PCM-16 Tukey-windowed linear up-sweep, emitted once every 12.5 ms. */
    fun pcm16(): ShortArray = ShortArray(frameSamples) { sample ->
        val t = sample.toDouble() / sampleRate
        val u = sample.toDouble() / frameSamples
        val phase = 2.0 * PI * (
            startHz * t + 0.5 * (endHz - startHz) / durationSeconds * t * t
        )
        (amplitude * tukey(u) * sin(phase) * Short.MAX_VALUE)
            .roundToInt()
            .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            .toShort()
    }

    private val durationSeconds: Double
        get() = frameSamples.toDouble() / sampleRate

    private fun tukey(u: Double): Double = when {
        u < tukeyAlpha / 2.0 -> 0.5 * (1.0 + cos(2.0 * PI / tukeyAlpha * (u - tukeyAlpha / 2.0)))
        u > 1.0 - tukeyAlpha / 2.0 -> 0.5 * (1.0 + cos(2.0 * PI / tukeyAlpha * (u - 1.0 + tukeyAlpha / 2.0)))
        else -> 1.0
    }
}

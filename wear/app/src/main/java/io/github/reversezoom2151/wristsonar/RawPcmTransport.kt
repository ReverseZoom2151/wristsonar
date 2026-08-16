package io.github.reversezoom2151.wristsonar

import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Writes the versioned binary format decoded by `wristsonar.capture.wire`. */
class RawPcmTransport(private val output: OutputStream) {
    fun send(samples: ShortArray, timestampNs: Long, sampleRate: Int) {
        require(samples.isNotEmpty() && samples.size <= 4_096) { "invalid callback size" }
        require(timestampNs >= 0) { "timestamp must be monotonic" }
        require(sampleRate > 0) { "sample rate must be positive" }
        val payload = ByteBuffer.allocate(19 + samples.size * 2)
            .order(ByteOrder.LITTLE_ENDIAN)
        payload.put(byteArrayOf('W'.code.toByte(), 'S'.code.toByte(), 'A'.code.toByte(), 'R'.code.toByte()))
        payload.put(1)
        payload.putInt(sampleRate)
        payload.putLong(timestampNs)
        payload.putShort(samples.size.toShort())
        samples.forEach(payload::putShort)
        output.write(payload.array())
        output.flush()
    }
}

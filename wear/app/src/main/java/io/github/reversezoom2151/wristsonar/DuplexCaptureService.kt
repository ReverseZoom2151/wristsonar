package io.github.reversezoom2151.wristsonar

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.os.IBinder
import android.util.Log
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.math.abs

/** Foreground service that owns the microphone while raw PCM goes to the host. */
class DuplexCaptureService : Service() {
    // Written on the transport thread and read by onDestroy on the main
    // thread. Without volatile, a service that has been killed can be left
    // holding the microphone because onDestroy never sees the engine.
    @Volatile
    private var engine: DuplexEngine? = null

    @Volatile
    private var socket: Socket? = null

    @Volatile
    private var streaming = false

    @Volatile
    private var writer: Thread? = null

    private val outbox = ArrayBlockingQueue<PcmFrame>(QUEUE_FRAMES)
    private var dropped = 0
    private var sent = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val host = intent?.getStringExtra(EXTRA_HOST) ?: return START_NOT_STICKY
        val port = intent.getIntExtra(EXTRA_PORT, -1)
        if (port !in 1..65_535) return START_NOT_STICKY
        startForeground(
            NOTIFICATION_ID,
            notification(),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
        )
        Thread({ connectAndCapture(host, port) }, "wristsonar-transport").start()
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        shutdown()
        super.onDestroy()
    }

    private fun connectAndCapture(host: String, port: Int) {
        try {
            val connected = Socket()
            // No connect timeout meant a wrong host IP hung this thread until
            // the kernel gave up, with the notification claiming capture was
            // active the whole time.
            connected.connect(InetSocketAddress(host, port), CONNECT_TIMEOUT_MILLIS)
            // Frames are 12.5 ms apart and small, so Nagle would coalesce them
            // into bursts and add latency the host reads as jitter.
            connected.tcpNoDelay = true
            connected.soTimeout = SOCKET_TIMEOUT_MILLIS
            connected.keepAlive = true
            socket = connected

            val transport = RawPcmTransport(connected.getOutputStream())
            streaming = true
            writer = Thread({ pump(transport) }, "wristsonar-writer").also { it.start() }

            val unprocessed = audioManager()
                .getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED)
                .toBoolean()
            Log.i(TAG, "connected to $host:$port, unprocessed source=$unprocessed")

            engine = DuplexEngine(
                unprocessedSupported = unprocessed,
                onFrame = ::enqueue,
                onFailure = { failure ->
                    Log.e(TAG, "capture failed", failure)
                    stopSelf()
                },
            ).also { it.start() }
        } catch (failure: Throwable) {
            // The old bare `catch (_: Throwable)` discarded every reason a
            // first bring-up can fail: no route, refused connection, denied
            // microphone, unsupported format. All of them looked identical.
            Log.e(TAG, "could not start the capture session", failure)
            shutdown()
            stopSelf()
        }
    }

    /**
     * Hand one frame to the writer thread, never to the network.
     *
     * This runs on the capture thread. A blocking socket write here stalls the
     * duplex loop for as long as the network wants, the record ring overflows,
     * and the samples the host assumes are contiguous quietly are not. Dropping
     * a frame instead is honest: it becomes a timestamp gap the host sees and
     * re-locks from, and it is counted and logged here.
     */
    private fun enqueue(pcm: ShortArray, timestampNs: Long, health: AudioHealth) {
        val effective = health.effectiveSampleRate
        if (effective != null) {
            val nominal = health.nominalSampleRate.toDouble()
            val measured = effective.toInt()
            check(abs(effective - nominal) <= nominal * RATE_TOLERANCE) {
                "device delivers $measured frames per second of boot time " +
                    "against a requested ${health.nominalSampleRate} Hz"
            }
        }
        check(health.agcEnabled != true && health.noiseSuppressorEnabled != true) {
            "capture path re-enabled AGC or noise suppression"
        }
        // A null AGC handle means gain control was not found in the effects
        // chain, which on most devices means it lives in the HAL where it can
        // be neither seen nor disabled. That is only acceptable on UNPROCESSED,
        // the one source defined to have none.
        check(health.unprocessedSource || health.agcEnabled == false) {
            "no controllable AGC on a source that is not UNPROCESSED, so gain " +
                "control across 18 to 21 kHz cannot be ruled out"
        }
        if (health.newOutputUnderruns > 0) {
            // Underruns shift the transmit cadence, which the host detects and
            // re-locks from. Killing the session on the first one, as this used
            // to, made a transient glitch permanent, and the old check could
            // never pass twice anyway because the platform counter is
            // cumulative and never resets.
            Log.w(TAG, "output underruns: ${health.newOutputUnderruns} new")
        }
        check(health.totalOutputUnderruns <= UNDERRUN_BUDGET) {
            "audio output underran ${health.totalOutputUnderruns} times; this " +
                "device cannot sustain duplex at ${health.nominalSampleRate} Hz"
        }
        if (!outbox.offer(PcmFrame(pcm, timestampNs))) {
            dropped += 1
            if (dropped % DROP_LOG_INTERVAL == 1) {
                Log.w(TAG, "transport behind, dropped $dropped frames so far")
            }
        }
    }

    private fun pump(transport: RawPcmTransport) {
        try {
            while (streaming) {
                val frame = outbox.poll(POLL_MILLIS, TimeUnit.MILLISECONDS) ?: continue
                transport.send(frame.samples, frame.startNanos, Fmcw.sampleRate)
                sent += 1
            }
        } catch (failure: Throwable) {
            Log.e(TAG, "transport stopped after $sent frames", failure)
            stopSelf()
        }
    }

    private fun shutdown() {
        streaming = false
        engine?.stop()
        engine = null
        // Closing before the join unblocks a writer parked in send().
        runCatching { socket?.close() }
            .onFailure { Log.w(TAG, "could not close the socket", it) }
        socket = null
        writer?.join(JOIN_MILLIS)
        writer = null
        Log.i(TAG, "session ended: $sent frames sent, $dropped dropped")
    }

    private fun audioManager(): AudioManager =
        getSystemService(Context.AUDIO_SERVICE) as AudioManager

    private fun notification(): Notification {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Wristsonar capture", NotificationManager.IMPORTANCE_LOW),
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Wristsonar capture active")
            .setContentText("Streaming raw microphone frames to the verified host")
            .setSmallIcon(android.R.drawable.presence_audio_online)
            .build()
    }

    private class PcmFrame(val samples: ShortArray, val startNanos: Long)

    companion object {
        const val EXTRA_HOST = "host"
        const val EXTRA_PORT = "port"
        private const val TAG = "WristsonarService"
        private const val CHANNEL_ID = "wristsonar.capture"
        private const val NOTIFICATION_ID = 1

        /** About 1.5 s of frames. Deep enough for a hiccup, not for a stall. */
        private const val QUEUE_FRAMES = 120
        private const val CONNECT_TIMEOUT_MILLIS = 5_000
        private const val SOCKET_TIMEOUT_MILLIS = 5_000
        private const val POLL_MILLIS = 200L
        private const val JOIN_MILLIS = 2_000L
        private const val DROP_LOG_INTERVAL = 40
        private const val RATE_TOLERANCE = 0.02
        private const val UNDERRUN_BUDGET = 50
    }
}

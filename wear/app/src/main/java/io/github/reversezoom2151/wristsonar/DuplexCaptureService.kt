package io.github.reversezoom2151.wristsonar

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import java.net.Socket

/** Foreground service that owns the microphone while raw PCM goes to the host. */
class DuplexCaptureService : Service() {
    private var engine: DuplexEngine? = null
    private var socket: Socket? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val host = intent?.getStringExtra(EXTRA_HOST) ?: return START_NOT_STICKY
        val port = intent.getIntExtra(EXTRA_PORT, -1)
        if (port !in 1..65_535) return START_NOT_STICKY
        startForeground(NOTIFICATION_ID, notification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        Thread({ connectAndCapture(host, port) }, "wristsonar-transport").start()
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        engine?.stop()
        runCatching { socket?.close() }
        super.onDestroy()
    }

    private fun connectAndCapture(host: String, port: Int) {
        try {
            val connected = Socket(host, port)
            socket = connected
            val transport = RawPcmTransport(connected.getOutputStream())
            engine = DuplexEngine(
                onFrame = { pcm, timestampNs, health ->
                    check(health.actualSampleRate == Fmcw.sampleRate) {
                        "device changed to ${health.actualSampleRate} Hz"
                    }
                    check(health.agcEnabled != true && health.noiseSuppressorEnabled != true) {
                        "capture path enabled AGC or noise suppression"
                    }
                    check(health.outputUnderruns == 0) { "audio output underrun" }
                    transport.send(pcm, timestampNs, Fmcw.sampleRate)
                },
                onFailure = { stopSelf() },
            ).also { it.start() }
        } catch (_: Throwable) {
            stopSelf()
        }
    }

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

    companion object {
        const val EXTRA_HOST = "host"
        const val EXTRA_PORT = "port"
        private const val CHANNEL_ID = "wristsonar.capture"
        private const val NOTIFICATION_ID = 1
    }
}

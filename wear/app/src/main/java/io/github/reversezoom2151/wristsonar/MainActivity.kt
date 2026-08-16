package io.github.reversezoom2151.wristsonar

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Toast

/** Minimal setup screen. The host address is explicit instead of silently guessed. */
class MainActivity : Activity() {
    private lateinit var host: EditText
    private lateinit var port: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        host = EditText(this).apply { hint = "Host IP" }
        port = EditText(this).apply { hint = "TCP port"; setText("8766") }
        val start = Button(this).apply {
            text = "Start capture"
            setOnClickListener { requestMicrophoneThenStart() }
        }
        val stop = Button(this).apply {
            text = "Stop"
            setOnClickListener { stopService(Intent(this@MainActivity, DuplexCaptureService::class.java)) }
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(host)
            addView(port)
            addView(start)
            addView(stop)
        })
    }

    private fun requestMicrophoneThenStart() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_MICROPHONE)
            return
        }
        val target = host.text.toString().trim()
        val targetPort = port.text.toString().toIntOrNull()
        if (target.isEmpty() || targetPort == null || targetPort !in 1..65_535) {
            Toast.makeText(this, "Enter a host IP and TCP port", Toast.LENGTH_SHORT).show()
            return
        }
        startForegroundService(Intent(this, DuplexCaptureService::class.java).apply {
            putExtra(DuplexCaptureService.EXTRA_HOST, target)
            putExtra(DuplexCaptureService.EXTRA_PORT, targetPort)
        })
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grants: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grants)
        if (requestCode == REQUEST_MICROPHONE && grants.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            requestMicrophoneThenStart()
        }
    }

    private companion object {
        const val REQUEST_MICROPHONE = 1
    }
}

# Wristsonar Wear OS sender

This Android module is the device half of the raw PCM protocol in
`wristsonar.capture.wire`. It starts 48 kHz mono duplex audio, sends a
Tukey-windowed 18 to 21 kHz chirp every 600 samples, and forwards unaligned
signed-16-bit microphone callbacks over TCP. The Python host synchronizes those
callbacks to chirp boundaries and produces the model input.

It deliberately refuses to stream when the device reports the wrong sample
rate, an enabled AGC or noise suppressor, or an AudioTrack underrun. Those are
measurement failures, not conditions a model is allowed to absorb.

## Build and validate

Open `wear/` in Android Studio with Android SDK 35 and a JDK supported by the
Android Gradle Plugin declared in `build.gradle.kts`. Install only on a Wear OS
watch with a speaker and microphone. The app has not been built or validated in
this repository's Linux environment because it has no JDK, Android SDK, or
physical watch.

Before interpreting a pose, validate all of the following on the target watch:

1. `AudioRecord.sampleRate` remains 48 kHz while the service is active.
2. The OS permits AGC and noise suppression to remain disabled for the selected
   source.
3. AudioTrack underruns remain zero during the capture interval.
4. Host `PcmSynchronizer` obtains and holds a chirp lock.
5. A live echo profile matches the WatchHand training product closely enough to
   justify a model run.

The sender uses `VOICE_RECOGNITION` because it is the least speech-processing
oriented public Android source. It is not proof of an unprocessed path. The
health checks and a device-specific signal comparison are what decide that.

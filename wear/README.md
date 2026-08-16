# Wristsonar Wear OS sender

This Android module is the device half of the raw PCM protocol in
`wristsonar.capture.wire`. It starts 48 kHz mono duplex audio, sends a
Tukey-windowed 18 to 21 kHz chirp every 600 samples, and forwards unaligned
signed-16-bit microphone callbacks over TCP. The Python host synchronizes those
callbacks to chirp boundaries and produces the model input.

## What the sender checks, and what it cannot

The sender refuses to stream when the capture path is measurably not the one
the model was trained on. Several of those checks are worth reading before
trusting a number that comes out of the host.

**Sample rate.** `AudioRecord.getSampleRate()` returns the rate that was
requested, not the rate the hardware runs at, so comparing it to 48000 compares
a constant to a constant and can never fail. The real check differences
`AudioRecord.getTimestamp()` frame positions against the boot-time clock over a
multi-second baseline, which recovers the rate frames actually arrive at. The
host repeats the same measurement independently from packet timestamps.

**Gain control.** The sender opens `UNPROCESSED` where
`AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED` says the device has
it, and falls back to `VOICE_RECOGNITION` otherwise. `VOICE_RECOGNITION` is
merely the least speech-processed of the remaining public sources, which is not
the same claim. `AutomaticGainControl.create()` returning null is not evidence
that no AGC is running: it usually means gain control lives in the HAL, where
it can be neither observed nor disabled. The service therefore treats a null
handle as acceptable only on `UNPROCESSED`. AGC across 18 to 21 kHz destroys
the amplitude relationships the differential echo profile is made of.

**Underruns.** `AudioTrack.getUnderrunCount()` is cumulative for the life of
the track and is never reset, so a single transient glitch used to fail every
subsequent frame and end the session permanently. The output ring is now
pre-rolled before `play()`, which removes the guaranteed startup underrun, and
the remaining underruns are logged as a delta and tolerated up to a budget. An
underrun shifts the transmit cadence, and the host's periodic correlation
re-lock is what actually deals with that.

**Timestamps.** A packet stamp is the capture time of the *first* sample in the
packet, on the boot-time monotonic clock. It comes from `AudioTimestamp` where
the device reports one, and otherwise from `elapsedRealtimeNanos()` back-dated
by one frame period. The host uses it only to spot gross gaps and to estimate
the effective rate; sample counting carries the alignment, because a stamp
taken after `read()` returns carries milliseconds of scheduling jitter and one
range bin is 20.8 microseconds.

**Transport.** The socket has a connect timeout, a read timeout and
`TCP_NODELAY`, and it is written from its own thread behind a bounded queue.
The capture thread never blocks on the network: a transport stall drops queued
frames, which the host sees as a gap and re-locks from, rather than overflowing
the record ring and silently deleting samples the host assumes are contiguous.

**Failures are logged.** Every failure path writes to logcat under
`WristsonarService` or `WristsonarDuplex`. Nothing is discarded silently.

## Build and validate

Open `wear/` in Android Studio with Android SDK 35 and a JDK supported by the
Android Gradle Plugin declared in `build.gradle.kts`. Install only on a Wear OS
watch with a speaker and microphone. The debug APK has been compiled locally
against Android SDK 35. It has not been installed or validated on a physical
watch, and the changes above have not been compiled since they were written.

The repository includes the Gradle wrapper, so a command-line build is:

```bash
cd wear
./gradlew :app:assembleDebug
```

Before interpreting a pose, validate all of the following on the target watch,
reading `adb logcat -s WristsonarService WristsonarDuplex`:

1. The service reports `unprocessed source=true`, or you have accepted that the
   fallback source may be applying gain control you cannot see.
2. The effective sample rate check does not fire, on the watch or on the host.
3. Underruns and dropped frames stay near zero over a full session.
4. Host `PcmSynchronizer` obtains and holds a chirp lock. The host prints its
   synchronisation reason, its re-lock count and its gap count when a session
   ends, and a session that emitted no pose frames says why.
5. A live echo profile matches the WatchHand training product closely enough to
   justify a model run.

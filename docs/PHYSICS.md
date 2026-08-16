# Physics

Why wrist-mounted active acoustic sensing can recover hand pose at all, why the
reported millimetre figures do not mean what they appear to mean, and why the
obvious cheaper alternative, listening to a passive impulsive sound, cannot work
for arm pose in any implementation.

This document is the load-bearing one. Everything in the evaluation harness
follows from it.

## What the device actually does

The watch speaker emits a linear frequency-modulated sweep, an FMCW chirp,
climbing from 18 kHz to 21 kHz over 12.5 ms, sampled and played back at 48 kHz.
Chirps are emitted back to back, so a frame arrives about every 12.5 ms, roughly
80 frames per second. The watch microphone records continuously through the same
period, which means the capture is duplex: the device is listening to its own
transmission and to everything that transmission bounces off.

The recorded frame is cross-correlated against the known transmitted chirp. The
correlation peak for a reflector at range r appears at the lag corresponding to
the acoustic path 2r, because sound travels out and back. Converting lag to
distance with c = 343 m/s gives a one-dimensional function of range: the echo
profile. This project's convention, stated in the signal layer, is that a range
always means the one-way reflector distance, with the factor of two already
absorbed into the metres-per-bin conversion.

Most of the energy in any single profile is the device itself: the direct
leakage path from speaker to microphone across the watch body, plus the static
reflection from the wrist the watch is strapped to. Subtracting consecutive
frames gives the differential profile, which suppresses everything that did not
move and leaves the fingers. The model sees both channels, the original and the
differential, cropped to the range window that can plausibly contain a hand and
stacked over about a second of frames. Cropping is not an optimisation. Bins
outside that window carry room reflections, and a model handed room reflections
will learn the room.

Differencing has a cost worth stating alongside its benefit: it is a high-pass
filter in time, so a hand held perfectly still becomes invisible in that
channel. These systems track motion and infer pose. They do not measure a
static hand.

## Why being on the body changes the problem

The same physics from a phone on a table is a much worse problem, and the
literature shows it: SonicHand on a stock Pixel 2 reports 23.25 mm within-user
and 39.28 mm cross-user, which its own authors call unsatisfactory. PoseSonic,
on custom smartglasses, gets 9 upper-body joints to 61.7 mm in the lab and
141.2 mm semi-in-the-wild.

The difference is not signal quality. It is the geometry of the unknowns.

A phone across the room sees an arbitrary hand at an arbitrary range, in an
arbitrary orientation, with a multipath structure that changes as the hand both
articulates and translates. Range, pose and orientation are entangled, and the
system has to disentangle them from a single scalar range profile.

A watch sits about 5 to 10 cm from the joints it is estimating, rigidly coupled
to the wrist, so that the relationship between sensor and joints is fixed up to
a mounting offset. The fingers cannot leave the sensor's field. Their range from
the device is bounded within a hand's length. Nothing in the scene competes for
the near bins, because the arm occludes the room behind it. And the quantity
that varies is exactly the quantity of interest: finger flexion changes the
range profile because it changes where the fingers are relative to a sensor that
moved with the wrist.

That also explains why the field's usual difficulties evaporate here. No
microphone array is required, because only range matters and there is nothing to
localise in angle. No true ultrasound is required, because 18 to 21 kHz is
sufficient at 5 to 10 cm. The operating system's beamformer and its multichannel
processing are irrelevant when there is one microphone and one dominant
reflector.

The cost of that geometry is that the mounting offset becomes part of the
signal. Take the watch off and put it back on and the occlusion geometry has
changed. That is not noise; it is a change of sensor. It is why cross-session
with remounting is the first split in this project that counts as a real number,
and why WatchHand degrades from 6.02 mm within-session to 7.87 mm across
sessions with remounting.

## The resolution argument

Here is the number that decides how to read every published result in this
field.

Range resolution for an FMCW system is c / 2B, where B is the swept bandwidth.
At the default 18 to 21 kHz sweep, B is 3 kHz, and

    343 / (2 x 3000) = 0.057 m

about 5.7 cm. That is the smallest separation at which two reflectors are
resolved as two reflectors. It is wider than a hand.

The plan document computes the same quantity with 4 kHz of usable inaudible
bandwidth and obtains 4.3 cm, and that is the figure it carries in its headline
argument. The code carries 5.7 cm, from the 18 to 21 kHz sweep the same plan
specifies for capture. The discrepancy is in the choice of B, not in the
physics, and it is unresolved: 4 kHz corresponds to an 18 to 22 kHz sweep, which
runs into the roll-off the code's own band selection is trying to avoid. Either
figure supports the argument, since both are one to two orders of magnitude
above the reported errors. This document uses 5.7 cm because it follows from the
chirp the project actually emits. Ring-a-Pose independently cites 34.3 mm as an
upper bound on conventional FMCW resolution, which is the same statement made
from a different bandwidth assumption.

There is a second limit, independent of bandwidth. The wavelength at the 19.5
kHz sweep centre is

    343 / 19500 = 0.0176 m

about 17.6 mm. Features much smaller than a wavelength do not reflect; they
diffract, and the scattered field carries almost nothing about their shape. A
fingertip is a few millimetres across. It is not an acoustic target at these
frequencies. What the sensor sees is the aggregate scattering of a hand-sized
object with hand-sized structure.

So when EchoWrist reports 4.81 mm mean per-joint error, and WatchHand reports
6.02 mm, those figures are between 3 and 12 times finer than the wavelength and
roughly 10 times finer than the range resolution. No amount of signal-to-noise
ratio closes that gap, because it is not a noise limit.

## What the millimetre numbers actually are

They are a learned prior, not resolved geometry.

The model is regressing a coarse echo signature onto the low-dimensional
manifold that human hands actually occupy. Twenty-one landmarks at three
coordinates each is sixty-three numbers, but tendon coupling and joint anatomy
mean a hand has far fewer effective degrees of freedom than that. Curl the ring
finger and the little finger follows. Most of the sixty-three-dimensional space
is anatomically unreachable. The measurement does not need to resolve a
fingertip to millimetres; it needs to identify which region of a much smaller
manifold the hand is in, and the manifold's own structure supplies the rest.

This is a description of the method, not an accusation. The systems work. They
are doing something legitimate and useful, and the accuracy they report is real
accuracy on the distribution they were tested on. The point is that the
mechanism is inference over a prior rather than measurement of geometry, and
that this changes what the numbers predict about deployment.

It predicts, and the literature observes, all of the following:

Excellent within-user accuracy, because the prior is fitted to that user's hand
and that user's mounting.

Two to three times degradation on unseen users. WatchHand goes from 7.87 mm
cross-session to 14.88 mm cross-user, leave-one-out.

Near-total immunity to ambient noise, which would be surprising for a
measurement and is expected for a prior: noise does not move the manifold.
EchoWrist went from 4.81 mm to 4.88 mm under injected music, with ANOVA p = 1.00.

Collapse on unseen gesture distributions, because interpolation inside a
training manifold says nothing about its edges. WatchHand's dynamic-transition
figure is 22.60 mm against 6.02 mm within-session.

Sensitivity to remounting, because the occlusion geometry is not a nuisance
parameter, it is a large part of the signal.

And it sets the ceiling. Performance is bounded by how well the training
distribution covers the poses that will actually be met, not by signal to noise
ratio. Which is why the evaluation harness, not the model, is the centre of this
project. See EVALUATION.md.

## Why passive impulsive sound cannot give arm pose

The tempting cheaper idea is to skip active sensing entirely and decode a
passive impulsive sound, a clap or a tap, into the pose of the arm that produced
it. This does not work, for four independent reasons, any one of which is
sufficient.

The sound is generated at and after contact. A clap radiates because air is
expelled from the collapsing cavity between the palms and excites a Helmholtz
resonance. Fu and colleagues (Physical Review Research 7, 013259, 2025)
validated that mechanism against high-speed imaging, engineered replicas and
finite element modelling.

The swing that precedes contact is acoustically silent. A hand moving at 5 to 10
m/s is at Mach 0.02. Dipole radiation from unsteady aerodynamic forces scales as
M^6, which puts the swing tens of decibels below room noise. Pre-impact
kinematics reach the microphone only insofar as they changed the state of the
system at the instant of contact.

Hand configuration and arm configuration are acoustically independent given the
contact state. The same cupped clap produced with the arms overhead, at the
chest, or behind the back yields the same direct-path spectrum. Room reflections
differ, but that is a statement about the room, not about the arm.

Even granting a microphone array and perfect localisation, what is recovered is
one 3D point: the place where the hands met. A human arm has seven degrees of
freedom. Constraining one endpoint of a 7-DOF chain leaves a four-dimensional
null space, the redundant-manipulator self-motion manifold, along which the
elbow can swing freely with the hand held still. Those four dimensions are not
poorly estimated. They are unobserved.

Finally the bit budget. A single impulse carries on the order of 5 to 10 bits,
of which roughly 1.5 to 2 concern the hand. A 7-DOF arm pose quantised to 5
degrees needs about 32 bits. That is short by a factor of 15 to 30. No model
architecture recovers information that was never in the channel.

The strongest empirical confirmation is PoseKernelLifter (CVPR 2022), which had
four speaker and microphone pairs, 96 kHz impulse responses, and a camera. Its
audio-only ablation is described by its own authors as highly erroneous, off by
more than an order of magnitude against the fused model.

What a clap does carry is hand configuration at the moment of contact, as a
coarse class. Repp (JASA 1987) established the taxonomy; Jylha and Erkut (DAFx
2008) reached 71.7 percent over 8 classes against 12.5 percent chance. That is
roughly 2 bits, and it is largely within-clapper. It is a legitimate small
result. It is not pose estimation, and this project does not attempt it. See
ROADMAP.md, phase 5, for where it might sit as a side quest.

## The commodity hardware ceilings

The constraints below are not engineering annoyances to be optimised away. They
are the boundary of what any system built on unmodified consumer hardware can
do, and they are why the band, the frame rate and the duty cycle are what they
are.

Sample rate is 48 kHz on every platform that matters, which puts Nyquist at 24
kHz. The chirp configuration rejects any sweep whose upper edge crosses that,
because it would alias. In practice the usable ceiling is lower still: consumer
speakers and microphones roll off hard above 21 to 22 kHz, and the anti-alias
filter takes what is left. The lower edge is set by human hearing, not by
physics, since below 18 kHz the sweep is audible to most people. Between those
two limits there are about 3 to 4 kHz to work with, and that number is the input
to the resolution argument above.

No platform gives raw multichannel access. Whatever the microphone array is
doing, the application sees the processed mono result. This project's design
depends on that not mattering, which it does not when there is one dominant
reflector at close range, but it does mean that any technique requiring array
geometry is unavailable by construction.

The operating system applies automatic gain control and noise suppression, and
on some platforms it will silently resample. AGC in particular is corrosive
here, because the differential profile is a difference of amplitudes and a
time-varying gain forges motion that did not happen. What can be done about it
is less than it sounds. The sender opens the `UNPROCESSED` source where the
device advertises support for it and falls back to `VOICE_RECOGNITION`
otherwise, which is only the least speech-processed of the remaining public
sources and is not the same claim. `AutomaticGainControl.create()` returning
null is not evidence that no gain control is running; it usually means the gain
control lives in the HAL, where it can be neither observed nor disabled. So
processing that has been applied anyway is, in general, undetectable from
inside the application.

Resampling is the one exception, and only because it can be timed rather than
inspected. A declared sample rate is useless here: `AudioRecord.getSampleRate()`
returns the rate that was requested, and a resampled stream reports the same
number on the wire, so comparing it to 48000 compares a constant to a constant.
Dividing the samples that actually arrived by elapsed monotonic time over a
multi-second baseline does not, and that measurement is made independently on
the watch from `AudioTimestamp` and on the host from packet timestamps. Nothing
else in this stack detects platform processing, and the documentation that used
to say otherwise was wrong.

Audible leakage is real. Across 30 commodity devices transmitting 18 to 22 kHz
at 80 percent volume, audible low-frequency leakage from amplifier nonlinearity
was present on most, exceeding 65 dB on some. Chirps leak less than other
waveforms, which is fortunate, but the figure has to be measured on the target
hardware rather than assumed.

Battery is the ceiling that turns a working system into a demo. Two hours of
continuous sensing drains 78 percent of a Galaxy Watch 3. Duty cycling is a
research problem in its own right and is not solved here.

Finally, transducer response in the 18 to 22 kHz band varies by tens of decibels
between device models. This is why cross-device is a distinct evaluation axis
from cross-user rather than a special case of it, and why it is the axis the
field usually does not report at all.

## Sources

WatchHand: arXiv 2602.21610, doi 10.1145/3772318.3790932,
github.com/witlab-kaist/WatchHand.
EchoWrist: arXiv 2401.17409, CHI 2024.
Ring-a-Pose: doi 10.1145/3699741, IMWUT 2024.
PoseSonic: scifilab.org/posesonic, IMWUT 2023.
SonicHand: engineering.purdue.edu/~lusu/papers/TOSN2024.pdf.
PoseKernelLifter: CVPR 2022, github.com/saic-ny/PoseKernelLifter.
Repp 1987: JASA 81(4):1100-1109.
Jylha and Erkut 2008: legacy.spa.aalto.fi/dafx08/papers/dafx08_52.pdf.
Fu et al. 2025: Physical Review Research 7, 013259.
PowerPhone (audible leakage across 30 devices): MobiCom 2023.

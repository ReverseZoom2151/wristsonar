# Security policy

## Supported versions

This project is pre-release and at version 0.0.x. Only the current `main`
branch receives fixes. There are no backports.

## Reporting a vulnerability

Please report privately rather than opening a public issue.

Email tibi.toca@gmail.com with a description of the issue, the steps or the
input needed to reproduce it, the commit or version you saw it on, and what an
attacker could do with it. If you prefer GitHub, use the repository's private
vulnerability reporting under the Security tab, which is also private until a
fix is published.

You can expect an acknowledgement within about a week. Once a fix is ready we
will credit you in the release notes unless you ask us not to. Please give us a
reasonable window to ship the fix before disclosing publicly.

Things worth reporting include anything that lets code or data escape the
process, anything that causes audio to be captured or retained outside the paths
the configuration describes, and anything that lets a crafted recording or model
file execute code during loading.

## What this project does with audio

This is not a conventional security concern, but it is the part of this project
most likely to harm someone, so it belongs here in plain terms.

wristsonar works by emitting sound from a smartwatch speaker in the 18 to 21 kHz
band and recording continuously from the same watch's microphone, then reading
hand pose out of the echoes. Two consequences follow, and both are real rather
than theoretical.

The emission is not silent to everyone. Hearing at 20 kHz declines with age, but
roughly half of adults under thirty can still detect tones at that frequency,
and some people perceive them as pressure, fatigue or discomfort rather than as
an audible tone. Anyone wearing the device, and ideally anyone sitting next to
it, should be told that it is emitting in that band and given a way to switch it
off. Do not describe the emission as inaudible in documentation, in a user
interface, or in a paper. Treat reported discomfort as a real effect rather than
as a suggestion.

A wrist-worn microphone recording continuously captures speech. It captures the
wearer's speech, the speech of everyone around the wearer, and it does so in
places where nobody has agreed to be recorded. The fact that the pipeline only
cares about ultrasound does not change what the microphone picked up, and it
does not change what a buffer, a crash dump or a debug recording can contain.
Any deployment must state where that audio goes, how long it is kept, whether it
leaves the device, who can reach it, and how it is deleted. If you cannot answer
those questions for your deployment, that is a finding, and this project would
rather hear about it early.

If you build on this, prefer discarding the audible band as close to the
microphone as possible, keep raw buffers out of logs and crash reports, and make
recording state visible to the wearer rather than implicit.

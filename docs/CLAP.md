# Clap configuration

What `wristsonar.clap` does, and more importantly what it does not.

## What it does

Given a recording containing an impulsive event, it detects the onset, fits a
low-order resonance model to the windowed impulse response, and classifies the
shape the hands were in at the moment they met. It also reports how many bits
of information that classification actually carries, so the component can state
its own limit rather than assert it.

That is the whole scope. A coarse contact-configuration class, a resonance
frequency and decay, an energy proxy, and an honest bit count.

## What it does not do

It does not estimate arm, wrist or finger positions, and no amount of model
capacity will change that. This component exists because a predecessor project
claimed exactly that, and the claim does not survive the physics.

Sound is generated at and after contact, by air expelled from the collapsing
cavity between the palms exciting a Helmholtz resonance. The swing that
produced the clap is acoustically silent: a hand at 5 to 10 m/s is roughly Mach
0.02, and dipole radiation scales as the sixth power of Mach number, which puts
the swing tens of dB below room noise. Whatever the arm did before contact
reaches the microphone only insofar as it changed the state at the instant of
contact.

Three consequences follow, each sufficient on its own.

Hand configuration and arm configuration are acoustically independent given the
contact state. The same cupped clap made with the arms overhead, at the chest,
or behind the back produces the same direct-path spectrum. The cavity is formed
between the two palms and its resonance is invariant to the entire proximal
chain.

Even granting a microphone array that localises the point where the hands met,
that yields one 3D point. Constraining a seven-degree-of-freedom arm to a single
endpoint leaves a four-dimensional null space, the self-motion manifold that
every redundant manipulator has.

And the information is simply not present. Running the capacity module on a
four-class confusion matrix at roughly the accuracy this literature reports
gives 0.73 bits. A perfect four-class classifier would give exactly 2.00 bits,
which is the ceiling for the merged taxonomy. Naming an arm pose at seven
degrees of freedom, over 120 degrees, at 5 degree steps costs 32.09 bits. The
realistic figure is short by a factor of about 44, and even the theoretical
ceiling is short by 16.

`assert_within_capacity` exists to make that check mechanical at the boundary
where a number leaves this package. It takes no tolerance parameter, on the
grounds that every over-claim in this literature was made by somebody who felt
their case was within tolerance.

## The taxonomy, and why it is four classes and not eight

Repp (1987) defined eight clapping modes: P1, P2 and P3 for parallel flat hands
running from palm-to-palm through to fingers-to-palm, and A1+, A1, A1-, A2 and
A3 for hands held at an angle, ordered by curvature and offset. That taxonomy
is preserved here as `ClapMode` for comparison with the literature.

It is not the default, because the acoustic evidence does not support eight
classes. Jylha and Erkut (2008) automated the classification and reported 71.7
percent on synthetic claps and about 64 percent on real ones from their best
subject, against 12.5 percent chance, with systematic confusions between two
specific pairs: P1 with A2, and P2 with A1. Those confusions are not noise.
They are the taxonomy asking for a distinction the signal does not make.

The default is therefore `ContactClass`, four merged classes:

| Class | Repp modes | Character |
| --- | --- | --- |
| `cupped-deep` | A1+ | Largest cavity, lowest resonance |
| `cupped-natural` | A1, P2 | Natural curvature |
| `flat` | A1-, P1, A2 | Little enclosed air |
| `fingers-to-palm` | A3, P3 | Smallest cavity, highest resonance |

Both reported confusable pairs are collapsed by that merge, which is the entire
argument for it. `CONFUSABLE_PAIRS` records them in code so the reasoning
survives. If a future dataset shows those pairs separating reliably, unmerge
them and say why. Do not unmerge them because four classes felt like too few.

## The physics the features rest on

The cavity between the palms behaves as a Helmholtz resonator, and this is now
well established rather than inferred. Fu et al. (Physical Review Research 7,
013259, 2025) validated the model against high-speed imaging of ten volunteers,
engineered replicas and finite element modelling. Cavity volume predicts clap
frequency, cavity gauge pressure scales with the square of closing speed, and
soft tissue damping is why a clap is short.

The resonance constrains the ratio of neck area to volume times effective
length, not volume alone. A large cavity with a large vent aliases onto a small
cavity with a small vent, so the implied volume this module reports is a
constrained quantity rather than a measurement.

`MODE_PROTOTYPES` carries centre frequencies from about 500 Hz for a very
cupped clap to about 1562 Hz for fingers-to-palm, following the range Peltola
et al. (2007) fitted for their per-clap-type resonators. Treat those as a
documented ordering and a sanity reference, not as calibration constants. No
two people's hands give the same numbers, and the useful information is the
ordering.

Repp also found no effect of hand size or sex on the clap spectrum, and sex
identification from claps ran at chance. That is worth knowing before anyone
proposes clapper identification as an application.

## Evaluating it honestly

Everything in the project's evaluation discipline applies here too, and the
capacity figure is not exempt. Bits are as protocol-dependent as accuracy, so
`capacity_measurement` returns a protocol-stamped `Measurement` rather than a
bare float. A within-session confusion matrix from a single clapper yields an
information figure that describes that clapper on that day.

The published results are a warning about scale. Jylha and Erkut's real-clap
numbers come from two subjects and forty claps, and Repp's listener result was
strong within a single clapper and much weaker across clappers. Nothing here
has been validated cross-subject, in noise, or at any scale worth the name.

## What it is legitimately good for

A clap is a good event trigger. It is loud, impulsive, easy to detect, and it
carries a couple of bits about the hand shape and a usable energy proxy. Used
that way, to fire a discrete action and perhaps select between a small number
of modes, it is honest and it works.

Described as pose estimation, it is not.

## References

Repp, "The sound of two hands clapping: an exploratory study", JASA
81(4):1100-1109, 1987.
Jylha and Erkut, "Inferring the hand configuration from hand clapping sounds",
DAFx 2008.
Peltola, Erkut, Cook and Valimaki, "Synthesis of hand clapping sounds", IEEE
TASLP 15(3):1021-1029, 2007.
Fu, Kiyama, Liu, Zhang and Jung, Physical Review Research 7, 013259, 2025.

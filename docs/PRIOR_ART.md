# Prior art

What exists in acoustic hand and body pose sensing, what it reports, under which
protocol, and what of it can actually be run.

The central observation, which is also the reason this project exists: there are
eight published systems in this family and zero runnable artifacts. No public
training code or weights exist for EchoWrist, Ring-a-Pose, PoseSonic,
EchoSpeech, RAM-Hand, SonicHand, Beyond-Voice, LeekyFeeder, or the
Meta-affiliated smart glasses work. The field releases datasets selectively and
models essentially never. A reader can obtain the numbers, sometimes the data,
and never the system.

## The systems

WatchHand (CHI 2026, KAIST WIT Lab with Cornell SciFi Lab) is the anchor. It
estimates 3D positions of finger joints using nothing but the built-in speaker
and microphone of an unmodified consumer smartwatch: Galaxy Watch 7, Xiaomi
Watch 2 Pro, Pixel Watch 3. FMCW 18 to 21 kHz at 48 kHz sampling, 12.5 ms chirps
giving about 80 frames per second, C-FMCW echo profiles fed to a FastViT-T12.
Forty participants, 35.6 hours, three watch brands, both wrists. Its dataset is
public under CC BY 4.0. Its reported figures, by protocol:

| Protocol | MPJPE |
|---|---|
| Within-session | 6.02 mm |
| Cross-session, with remounting | 7.87 mm |
| Cross-user, leave-one-out | 14.88 mm |
| Dynamic transitions | 22.60 mm |

WatchHand is unusual in this literature precisely because it reports all four.
The ratio between the first and the third, roughly 2.5 times, is the quantity
this project's evaluation harness is designed to make impossible to omit.

The rest of the field, with the figures each reports:

| System | Venue | Hardware | Reported accuracy | Protocol notes |
|---|---|---|---|---|
| WatchHand | CHI 2026 | Stock smartwatch, 3 brands | 6.02 / 7.87 / 14.88 / 22.60 mm | Four protocols reported, table above. 40 participants, 35.6 h |
| EchoWrist | CHI 2024 | Not stated in the plan | 4.81 mm | Also 4.88 mm under injected music, ANOVA p = 1.00. Calibration curve 12.2 mm cold to 6.92 mm after one minute, plateau at twenty |
| Ring-a-Pose | IMWUT 2024 | Not stated in the plan | Cites 34.3 mm as the upper bound on conventional FMCW range resolution | The bound is the citation worth keeping |
| PoseSonic | IMWUT 2023 | Custom smartglasses | 61.7 mm lab, 141.2 mm semi-in-the-wild | 9 upper-body joints, not hand pose |
| SonicHand | TOSN 2024 | Stock Pixel 2 | 23.25 mm within-user, 39.28 mm cross-user | Authors call the cross-user figure unsatisfactory |
| PoseKernelLifter | CVPR 2022 | 4 speaker and mic pairs, 96 kHz impulse responses, plus a camera | Audio-only ablation described by its authors as highly erroneous, off by more than an order of magnitude against the fused model | The strongest evidence against audio-only body pose |
| RAM-Hand | SenSys 2025 | Not stated in the plan | Not extracted here | No artifact |
| LeekyFeeder | Google Research | In-air gesture control through leaky acoustic waves | Not extracted here | No artifact |

EchoSpeech, Beyond-Voice and the Meta-affiliated smart glasses work belong to
the same family and are listed among the systems with no public code or weights,
but their accuracy figures are not extracted in the plan and are not invented
here.

Two figures worth carrying from outside the acoustic literature, as the standard
this field is measured against rather than as competition: MediaPipe Hand
Landmarker is free and runs in 17.1 ms on a Pixel 6 CPU, and WiLoR (CVPR 2025)
reaches 5.5 mm PA-MPJPE. The defensible framing for a wrist-mounted acoustic
system is form factor and line of sight, not accuracy: it sees the hand when no
camera can, all day, at tens of milliwatts.

## What is actually released

| Artifact | Contents | Licence |
|---|---|---|
| github.com/witlab-kaist/WatchHand | 35.6 h, 40 participants, 3 watch brands | Paper CC BY 4.0 |
| github.com/cjlisalee/Grab-n-Go_Data | 20,160 C-FMCW microgesture instances, 18 participants, echo profiles as .npy, full DSP recipe | Cornell eCommons, doi 10.7298/7kbd-vv75 |
| github.com/saic-ny/PoseKernelLifter | 10,000+ poses, 6 environments, 96 kHz room impulse responses | Other |
| github.com/YutoShibata07/AcousticPose_Public | CVPR 2023 acoustic body pose, training and eval code | Other |
| github.com/Samsonsjarkal/LLAP | iOS 17.5 to 20.3 kHz phase tracking, 3.5 mm 1D, 15 ms latency | MIT |
| github.com/wanganran/MilliSonic | Server plus 4-mic Arduino firmware | Apache-2.0 |
| github.com/leeyadong/UltraPoser | Commodity multi-device acoustic plus IMU full body | Unconfirmed |

Read that table against the previous one. Every entry is a dataset, a signal
processing reference, or a body-pose system from an adjacent problem. None is a
runnable hand-pose model on commodity hardware. The two permissively licensed
entries, LLAP under MIT and MilliSonic under Apache-2.0, are signal-layer
references rather than pose systems, and LLAP in particular is the reference
implementation for duplex phase tracking on a phone and is worth reading before
writing any capture code. Two of the most relevant artifacts carry a licence
listed only as Other, which has to be resolved before any code derived from them
ships.

## Why reimplementation is feasible

The signal chain is fully specified in prose across the EchoWrist, Ring-a-Pose
and WatchHand papers, including filter orders, chirp lengths, window sizes and
crop geometry. The Grab-n-Go README documents echo-profile computation end to
end. Reimplementation is real work, but it is engineering work, not research
risk. The gap in this field is not knowledge. It is that nobody has shipped the
artifact.

## The passive-sound literature

Kept separate because it addresses a different question and is used in
PHYSICS.md to close off a direction rather than to open one.

Repp (JASA 1987, 81(4):1100-1109) established the taxonomy of hand
configurations in clapping. Jylha and Erkut (DAFx 2008) classified 8 of those
configurations at 71.7 percent against 12.5 percent chance. Fu and colleagues
(Physical Review Research 7, 013259, 2025) validated the Helmholtz resonance
mechanism against high-speed imaging, engineered replicas and finite element
modelling. Together these establish that a clap carries roughly 2 bits about
hand configuration, largely within-clapper, and nothing about arm pose.

## Adjacent results that bound the claims

The knee-acoustics reproduction study (arXiv 2405.15085) found 96 percent
leave-one-session-out accuracy from a single healthy subject recorded over five
days with no pathology present. It is the reason this project's harness treats
weak splits as adversarial rather than as a starting point. See EVALUATION.md.

On the clinical side, rejection rates for upper-limb prostheses have not moved
in 25 years, at 23 percent myoelectric, 26 percent body-powered and 39 percent
passive, and pattern recognition showed no functional advantage over direct
control on 74 percent of metrics. CMS reimbursement code L6700 specifies EMG
inputs. Sonomyography, the closest acoustic analogue, has 12 years of NIH
funding and a running trial and remains investigational. This project makes no
prosthetics claims.

The Meta neuromotor interface (Nature 645:702, 2025, doi
10.1038/s41586-025-09255-w) is the strongest recent result in wrist-worn input
generally, and is a useful reference point for what a wrist band can do with a
different sensing modality.

## Sources

WatchHand: arXiv 2602.21610, doi 10.1145/3772318.3790932,
github.com/witlab-kaist/WatchHand.
EchoWrist: arXiv 2401.17409, CHI 2024.
Ring-a-Pose: doi 10.1145/3699741, IMWUT 2024.
PoseSonic: scifilab.org/posesonic, IMWUT 2023.
SonicHand: engineering.purdue.edu/~lusu/papers/TOSN2024.pdf.
LeekyFeeder:
research.google/pubs/leakyfeeder-in-air-gesture-control-through-leaky-acoustic-waves/.
RAM-Hand: wenjunjiang.github.io/SenSys2025Shiyang.pdf.
PoseKernelLifter: CVPR 2022, github.com/saic-ny/PoseKernelLifter.
Repp 1987: JASA 81(4):1100-1109.
Jylha and Erkut 2008: legacy.spa.aalto.fi/dafx08/papers/dafx08_52.pdf.
Fu et al. 2025: Physical Review Research 7, 013259.
PowerPhone: MobiCom 2023.
Knee acoustics reproduction study: arXiv 2405.15085.
Meta neuromotor interface: Nature 645:702 (2025), doi 10.1038/s41586-025-09255-w.
MediaPipe Hand Landmarker:
developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker.
WiLoR: CVPR 2025.
LLAP: github.com/Samsonsjarkal/LLAP (MIT).

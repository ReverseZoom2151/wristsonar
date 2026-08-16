"""Clap configuration classes, and the decision to stop resolving them.

Repp (1987) catalogued eight ways two hands meet: P1, P2 and P3 for parallel
flat hands moving from palm-to-palm through to fingers-to-palm, A1, A2 and A3
for hands held at an angle with natural curvature, plus A1+ for a very cupped
clap and A1- for a very flat one. Palm-to-palm gives a narrow resonance below
1 kHz with a notch near 2.5 kHz; fingers-to-palm gives a broad peak near
2 kHz. Smaller enclosed cavity, higher resonance. Repp also found no effect
of hand size or sex on the spectrum, and listeners identified the clapper's
sex at chance, which is worth remembering whenever someone proposes a
biometric built on this.

The design decision this module makes
-------------------------------------
Eight classes is more resolution than the acoustics support, and the evidence
for that is in the paper that tried hardest to make eight work. Jylha and
Erkut (DAFx 2008) fitted a second-order all-pole model by Steiglitz-McBride
and classified from its coefficients, reaching 71.7 percent on synthetic
claps and about 64 percent on real claps from their better subject, against
12.5 percent chance. Two features of that result matter more than the
headline number:

- A1+ ran above 90 percent. A very cupped clap has a distinctly larger
  cavity, so its resonance sits far below everything else, near 500 Hz in
  Peltola et al. (2007), and it separates cleanly.
- The errors were not diffuse. They concentrated in two specific pairs, P1
  confused with A2 and P2 confused with A1. Those are not random slips. They
  are pairs whose enclosed cavities are close in size, so they are close in
  the only two numbers a second-order fit produces.

A taxonomy whose distinctions are systematically unrecoverable is not a
taxonomy, it is a source of confident wrong answers. So this module merges
along the confusion structure and defaults to the merged scheme:

    CUPPED_DEEP      A1+                the one class the evidence isolates
    CUPPED_NATURAL   A1, P2             the P2-to-A1 confusion pair
    FLAT             A1-, P1, A2        the P1-to-A2 pair plus flat A1-,
                                        whose prototype frequency sits
                                        between them
    FINGERS_TO_PALM  A3, P3             the high, broad, small-cavity end

Four classes, chance 25 percent instead of 12.5 percent, and the merges
absorb exactly the errors the literature reports rather than being chosen for
convenience. The eight-mode scheme is kept, selectable, so results can be
compared against Repp and against Jylha and Erkut directly. It is not the
default because defaults are what people ship.

What is deliberately not a feature
----------------------------------
Loudness. Fu et al. (2025) showed cavity gauge pressure scales quadratically
with clap speed, so energy tracks how hard someone clapped. Letting it into
the classifier would buy accuracy on any dataset where subjects clapped
harder in some configurations than others, and would lose it on any dataset
where they did not. :class:`ClapFeatures` carries energy so callers can use
it as the amplitude proxy it is, and :meth:`ClapFeatures.vector` leaves it
out.

The feature vector is two dimensional because a second-order all-pole model
has two free parameters after gain: pole angle and pole radius, which is to
say centre frequency and bandwidth. Adding decay rate as a third feature
would be adding bandwidth again in different units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from wristsonar.clap.resonance import ResonanceFit
from wristsonar.protocol import Measurement, Protocol

__all__ = [
    "CONFUSABLE_PAIRS",
    "DEFAULT_TAXONOMY",
    "MERGE_MAP",
    "MODE_PROTOTYPES",
    "ClapFeatures",
    "ClapMode",
    "ConfigurationClassifier",
    "ContactClass",
    "Label",
    "Taxonomy",
    "accuracy",
    "accuracy_measurement",
    "features_from_fit",
    "merge_mode",
    "prototype_classifier",
]

_EPS = 1e-12


class ClapMode(Enum):
    """Repp's eight modes, kept for comparison with the literature."""

    P1 = "P1"
    """Parallel flat hands, palm to palm."""

    P2 = "P2"
    """Parallel flat hands, palm to palm shifted toward the fingers."""

    P3 = "P3"
    """Parallel flat hands, fingers to palm."""

    A1PLUS = "A1+"
    """Hands at an angle, very cupped. The largest cavity, the lowest note."""

    A1 = "A1"
    """Hands at an angle, natural curvature, palm to palm."""

    A1MINUS = "A1-"
    """Hands at an angle, very flat."""

    A2 = "A2"
    """Hands at an angle, part way toward the fingers."""

    A3 = "A3"
    """Hands at an angle, fingers to palm."""


class ContactClass(Enum):
    """The merged taxonomy the acoustic evidence supports."""

    CUPPED_DEEP = "cupped-deep"
    CUPPED_NATURAL = "cupped-natural"
    FLAT = "flat"
    FINGERS_TO_PALM = "fingers-to-palm"


class Taxonomy(Enum):
    """Which label set a classifier works in."""

    EIGHT_MODE = "eight-mode"
    """Repp's eight. Over-resolved. Present for comparison only."""

    MERGED = "merged"
    """Four classes, merged along the reported confusion structure."""


DEFAULT_TAXONOMY = Taxonomy.MERGED

MERGE_MAP: dict[ClapMode, ContactClass] = {
    ClapMode.A1PLUS: ContactClass.CUPPED_DEEP,
    ClapMode.A1: ContactClass.CUPPED_NATURAL,
    ClapMode.P2: ContactClass.CUPPED_NATURAL,
    ClapMode.A1MINUS: ContactClass.FLAT,
    ClapMode.P1: ContactClass.FLAT,
    ClapMode.A2: ContactClass.FLAT,
    ClapMode.A3: ContactClass.FINGERS_TO_PALM,
    ClapMode.P3: ContactClass.FINGERS_TO_PALM,
}

CONFUSABLE_PAIRS: tuple[tuple[ClapMode, ClapMode], ...] = (
    (ClapMode.P1, ClapMode.A2),
    (ClapMode.P2, ClapMode.A1),
)
"""The pairs Jylha and Erkut reported as systematically confused.

Both are collapsed by :data:`MERGE_MAP`, which is the whole argument for the
merge. If a future dataset shows these separating reliably, unmerge them and
say why; do not unmerge them because four classes felt like too few.
"""

MODE_PROTOTYPES: dict[ClapMode, tuple[float, float]] = {
    ClapMode.A1PLUS: (500.0, 110.0),
    ClapMode.A1: (794.0, 180.0),
    ClapMode.P2: (850.0, 130.0),
    ClapMode.A1MINUS: (1010.0, 210.0),
    ClapMode.P1: (1063.0, 150.0),
    ClapMode.A2: (1120.0, 240.0),
    ClapMode.A3: (1350.0, 300.0),
    ClapMode.P3: (1562.0, 380.0),
}
"""Prototype (centre frequency Hz, bandwidth Hz) per mode.

Centre frequencies follow the range Peltola et al. (2007) fitted for their
per-clap-type resonators, from about 500 Hz for a very cupped clap to about
1562 Hz for fingers to palm, ordered by decreasing cavity size as Repp's
account requires. Bandwidths are narrower for the parallel modes, following
Repp's observation that palm-to-palm produces a narrow resonance while the
fingers-to-palm end is broad. Treat these as a documented ordering and a
sanity reference, not as calibration constants: no two people's hands give
the same numbers, and the useful information is the ordering, not the value.
"""

Label = ClapMode | ContactClass


def merge_mode(mode: ClapMode) -> ContactClass:
    """Map one of Repp's modes onto the merged class it belongs to."""
    return MERGE_MAP[mode]


def _labels_for(taxonomy: Taxonomy) -> tuple[Label, ...]:
    if taxonomy is Taxonomy.EIGHT_MODE:
        return tuple(ClapMode)
    return tuple(ContactClass)


def _project(label: Label, taxonomy: Taxonomy) -> Label:
    """Express a label in the requested taxonomy."""
    if taxonomy is Taxonomy.EIGHT_MODE:
        if isinstance(label, ContactClass):
            raise ValueError(
                "a merged class cannot be expanded back into one of Repp's "
                "eight modes; the merge is deliberately lossy"
            )
        return label
    return merge_mode(label) if isinstance(label, ClapMode) else label


@dataclass(frozen=True, slots=True)
class ClapFeatures:
    """What one impulse offers a classifier.

    Two numbers, plus an energy that is carried but not classified on. See
    the module docstring for why the vector is two dimensional and why
    energy stays out of it.
    """

    centre_frequency_hz: float
    bandwidth_hz: float
    energy: float = 0.0

    def vector(self) -> NDArray[np.float64]:
        """Log-scaled feature vector.

        Logs because both quantities are ratio-scaled: the perceptually and
        physically meaningful difference between 500 and 600 Hz is the same
        as between 1000 and 1200 Hz, and Helmholtz volume goes as 1/f^2, so a
        log frequency axis is linear in log volume.
        """
        return np.array(
            [
                math.log10(max(self.centre_frequency_hz, _EPS)),
                math.log10(max(self.bandwidth_hz, _EPS)),
            ],
            dtype=np.float64,
        )


def features_from_fit(fit: ResonanceFit) -> ClapFeatures:
    """Read features off a resonator fit."""
    return ClapFeatures(
        centre_frequency_hz=fit.centre_frequency_hz,
        bandwidth_hz=fit.bandwidth_hz,
        energy=fit.energy,
    )


@dataclass(frozen=True, slots=True)
class ConfigurationClassifier:
    """A shared-covariance Gaussian classifier over two log features.

    Deliberately the simplest thing that can work. With two features and a
    handful of classes there is nothing for a larger model to learn except
    the recording conditions, and this literature's failure mode is exactly
    that: high accuracy on one subject in one room, collapse elsewhere. A
    diagonal, pooled-variance Gaussian has one mean per class and two
    variances in total, so it cannot memorise a room.
    """

    taxonomy: Taxonomy
    labels: tuple[Label, ...]
    means: NDArray[np.float64]
    """Shape (n_labels, 2)."""

    variances: NDArray[np.float64]
    """Shape (2,). Pooled within-class variance."""

    def __post_init__(self) -> None:
        if self.means.shape != (len(self.labels), 2):
            raise ValueError(
                f"expected means of shape {(len(self.labels), 2)}, "
                f"got {self.means.shape}"
            )
        if self.variances.shape != (2,):
            raise ValueError("variances must have shape (2,)")
        if np.any(self.variances <= 0.0):
            raise ValueError("variances must be positive")

    @classmethod
    def fit(
        cls,
        features: list[ClapFeatures],
        labels: list[Label],
        *,
        taxonomy: Taxonomy = DEFAULT_TAXONOMY,
        variance_floor: float = 1e-6,
    ) -> ConfigurationClassifier:
        """Fit class means and a pooled within-class variance."""
        if len(features) != len(labels):
            raise ValueError("features and labels must be the same length")
        if not features:
            raise ValueError("cannot fit a classifier on no data")

        projected = [_project(label, taxonomy) for label in labels]
        vectors = np.stack([f.vector() for f in features])
        present = _labels_for(taxonomy)

        means = np.zeros((len(present), 2), dtype=np.float64)
        residuals: list[NDArray[np.float64]] = []
        for index, label in enumerate(present):
            mask = np.array([p is label for p in projected], dtype=bool)
            if not mask.any():
                raise ValueError(f"no training examples for label {label}")
            group = vectors[mask]
            means[index] = group.mean(axis=0)
            residuals.append(group - means[index])

        pooled = np.concatenate(residuals, axis=0)
        variances = np.maximum(pooled.var(axis=0), variance_floor)
        return cls(
            taxonomy=taxonomy,
            labels=present,
            means=means,
            variances=variances,
        )

    def scores(self, features: ClapFeatures) -> NDArray[np.float64]:
        """Log likelihood per label, up to a shared constant."""
        delta = features.vector()[None, :] - self.means
        return -0.5 * np.sum(np.square(delta) / self.variances, axis=1)

    def predict(self, features: ClapFeatures) -> Label:
        return self.labels[int(np.argmax(self.scores(features)))]

    def predict_proba(self, features: ClapFeatures) -> NDArray[np.float64]:
        """Posterior over labels under a uniform prior."""
        scores = self.scores(features)
        shifted = np.exp(scores - float(np.max(scores)))
        return np.asarray(shifted / float(np.sum(shifted)), dtype=np.float64)

    def confusion_matrix(
        self, features: list[ClapFeatures], labels: list[Label]
    ) -> NDArray[np.int64]:
        """Counts of true label (rows) against predicted label (columns).

        Row and column order is :attr:`labels`.
        """
        if len(features) != len(labels):
            raise ValueError("features and labels must be the same length")
        index = {label: i for i, label in enumerate(self.labels)}
        matrix = np.zeros((len(self.labels), len(self.labels)), dtype=np.int64)
        for feature, label in zip(features, labels, strict=True):
            true = _project(label, self.taxonomy)
            matrix[index[true], index[self.predict(feature)]] += 1
        return matrix

    @property
    def chance_accuracy(self) -> float:
        """Accuracy of guessing uniformly at random."""
        return 1.0 / len(self.labels)


def prototype_classifier(
    taxonomy: Taxonomy = DEFAULT_TAXONOMY,
    *,
    log_frequency_sd: float = 0.035,
    log_bandwidth_sd: float = 0.13,
) -> ConfigurationClassifier:
    """A classifier built from :data:`MODE_PROTOTYPES` with no training data.

    Useful as a zero-data baseline and as a way to see the geometry of the
    problem, since the merged class means are just the prototype means of
    their members. The default spreads correspond to roughly 8 percent
    frequency variation and roughly 35 percent bandwidth variation between
    claps of the same configuration, which is the order of within-class
    scatter the literature reports. It will not beat a classifier fitted on
    the target speaker and hands, and it is not meant to.
    """
    present = _labels_for(taxonomy)
    means = np.zeros((len(present), 2), dtype=np.float64)
    for index, label in enumerate(present):
        if isinstance(label, ClapMode):
            members = [label]
        else:
            members = [m for m, c in MERGE_MAP.items() if c is label]
        vectors = np.stack(
            [ClapFeatures(*MODE_PROTOTYPES[member]).vector() for member in members]
        )
        means[index] = vectors.mean(axis=0)
    variances = np.array([log_frequency_sd**2, log_bandwidth_sd**2], dtype=np.float64)
    return ConfigurationClassifier(
        taxonomy=taxonomy, labels=present, means=means, variances=variances
    )


def accuracy(confusion: NDArray[np.int64]) -> float:
    """Fraction correct from a confusion matrix."""
    total = int(np.sum(confusion))
    if total == 0:
        raise ValueError("confusion matrix is empty")
    return float(np.trace(confusion)) / total


def accuracy_measurement(
    confusion: NDArray[np.int64], protocol: Protocol
) -> Measurement:
    """Wrap a classification accuracy in the protocol that produced it.

    Accuracy without a protocol is not a result in this project. A
    within-session number from one clapper in one room says nothing about
    whether the classifier works, and :class:`~wristsonar.protocol.Protocol`
    makes that visible rather than leaving it to a caption.
    """
    return Measurement(
        value=accuracy(confusion),
        unit="fraction correct",
        protocol=protocol,
        samples=int(np.sum(confusion)),
    )

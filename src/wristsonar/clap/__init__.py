"""Coarse contact-configuration sensing from impulsive hand sounds.

This package is the one piece of the predecessor project, Dapper, that
survived scrutiny. Dapper claimed to predict 3D arm, wrist and finger
positions from a passive dap. That claim fails for physical reasons, not
engineering ones, and the reasons are set out in docs/CLAP.md and quantified
by :mod:`wristsonar.clap.capacity`.

What is left after the claim is removed is still worth having. A clap is a
Helmholtz resonator being excited once, and the resonance frequency of that
resonator is set by the cavity the hands enclose at the instant of contact.
So a single impulse supports a coarse class label for hand configuration,
plus an energy proxy that tracks clap speed. It does not support a pose.

Modules:

- :mod:`~wristsonar.clap.detect` finds impulsive events and rejects
  sustained sound.
- :mod:`~wristsonar.clap.resonance` fits a low-order all-pole model and
  reports the dominant resonance, with a Helmholtz reading of it.
- :mod:`~wristsonar.clap.configuration` classifies the configuration, in
  both Repp's eight-mode taxonomy and the merged taxonomy the acoustic
  evidence actually supports.
- :mod:`~wristsonar.clap.capacity` states the information limit in bits and
  refuses to let a caller claim past it.
"""

from __future__ import annotations

from wristsonar.clap.capacity import (
    ARM_DOF,
    ARM_JOINT_RANGE_DEG,
    HAND_BITS_RANGE,
    IMPULSE_BITS_RANGE,
    POSE_RESOLUTION_DEG,
    ChannelCapacity,
    OverclaimError,
    PoseGap,
    arm_pose_bits,
    assert_within_capacity,
    capacity_measurement,
    channel_capacity,
    mutual_information_bits,
    pose_gap,
)
from wristsonar.clap.configuration import (
    CONFUSABLE_PAIRS,
    DEFAULT_TAXONOMY,
    MERGE_MAP,
    MODE_PROTOTYPES,
    ClapFeatures,
    ClapMode,
    ConfigurationClassifier,
    ContactClass,
    Label,
    Taxonomy,
    accuracy,
    accuracy_measurement,
    features_from_fit,
    merge_mode,
    prototype_classifier,
)
from wristsonar.clap.detect import ImpulseEvent, detect_impulses
from wristsonar.clap.resonance import (
    ResonanceFit,
    ResonanceFitError,
    cavity_volume_m3,
    fit_resonance,
    helmholtz_frequency_hz,
    helmholtz_geometry_ratio,
    steiglitz_mcbride_fit,
)

__all__ = [
    "ARM_DOF",
    "ARM_JOINT_RANGE_DEG",
    "CONFUSABLE_PAIRS",
    "DEFAULT_TAXONOMY",
    "HAND_BITS_RANGE",
    "IMPULSE_BITS_RANGE",
    "MERGE_MAP",
    "MODE_PROTOTYPES",
    "POSE_RESOLUTION_DEG",
    "ChannelCapacity",
    "ClapFeatures",
    "ClapMode",
    "ConfigurationClassifier",
    "ContactClass",
    "ImpulseEvent",
    "Label",
    "OverclaimError",
    "PoseGap",
    "ResonanceFit",
    "ResonanceFitError",
    "Taxonomy",
    "accuracy",
    "accuracy_measurement",
    "arm_pose_bits",
    "assert_within_capacity",
    "capacity_measurement",
    "cavity_volume_m3",
    "channel_capacity",
    "detect_impulses",
    "features_from_fit",
    "fit_resonance",
    "helmholtz_frequency_hz",
    "helmholtz_geometry_ratio",
    "merge_mode",
    "mutual_information_bits",
    "pose_gap",
    "prototype_classifier",
    "steiglitz_mcbride_fit",
]

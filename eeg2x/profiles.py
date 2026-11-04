"""Versioned, model-independent preprocessing profiles."""
from __future__ import annotations

from eeg2x.protocol import PreprocessingProfile, ProcessingStep


def _p(profile_id, task_id, *steps, version="1.0.0"):
    return PreprocessingProfile(
        profile_id=profile_id, task_id=task_id,
        steps=tuple(ProcessingStep(name=n, fit_scope=s, parameters=v)
                    for n, s, v in steps),
        version=version,
    )


READ_BV = ("read_brainvision", "none", {"type": "eeg", "units": "uV"})
RS250 = ("resample_poly", "none", {"target_hz": 250.0})
RS200 = ("resample_poly", "none", {"target_hz": 200.0})
RS256 = ("resample_poly", "none", {"target_hz": 256.0})
IMAGE_EPOCH = ("epoch", "none", {"t_pre_sec": 0.0, "t_post_sec": 1.0})
MNN = ("mnn_whiten", "train", {
    "covariance": "ledoit_wolf", "output_units": "dimensionless_whitened"})

# ATM generation follows the paper's THINGS-EEG2 preprocessing. Intervals are
# half-open so the declared 250 Hz output has exactly 250 post-stimulus samples.
ATM_IMAGE_EPOCH = ("epoch", "none", {
    "interval_sec": [-0.2, 1.0], "interval_semantics": "half_open",
    "event_source": "stim_trigger_rising_edge",
})
ATM_BASELINE = ("baseline", "none", {
    "interval_sec": [-0.2, 0.0], "interval_semantics": "half_open",
    "operation": "subtract_channel_mean",
})
ATM_IMAGE_CROP = ("crop", "none", {
    "interval_sec": [0.0, 1.0], "interval_semantics": "half_open",
    "n_samples": 250,
})
REVE_IMAGE_CROP = ("crop", "none", {
    "interval_sec": [0.0, 1.0], "interval_semantics": "half_open",
    "n_samples": 200,
})
LABRAM_IMAGE_CROP = ("crop", "none", {
    "interval_sec": [0.0, 1.0], "interval_semantics": "half_open",
    "n_samples": 200,
})

# NOD image generation reads the dataset's own released clean continuous files
# (250 Hz, already 0.1-100 Hz band-passed, bad-channel interpolated, Zapline
# 50-Hz cleaned, average-referenced and ICA-regressed by the dataset authors).
# Nothing is re-filtered here: the only THINGS-EEG alignment kept for NOD is the
# 0.1-100 Hz band and the 250 Hz clock, both already satisfied by those files.
# Epoching follows the NOD paper's -100..800 ms window; every encoder then
# applies its own sampling-rate and zero-padding adaptation.
NOD_READ_CLEAN = ("read_dataset_clean_continuous", "none", {
    "source": "NOD-EEG/derivatives/preprocessed/raw/"
              "<sub>_ses-ImageNetNN_task-ImageNet_run-NN_eeg_clean.fif",
    "source_format": "MNE_FIF_continuous",
    "sampling_rate_hz": 250.0,
    "dataset_provided_preprocessing": [
        "0.1-100 Hz band-pass",
        "RANSAC bad-channel detection with spherical-spline interpolation",
        "50-Hz Zapline (line noise and harmonics)",
        "average EEG reference",
        "ICA artifact regression (fit 1-100 Hz, applied to the 0.1-100 Hz series)",
    ],
    "already_applied_by_dataset": True,
    "derived_inputs_used": False,
})
NOD_EPOCH = ("epoch", "none", {
    "interval_sec": [-0.1, 0.8], "interval_semantics": "mne_tmin_tmax_inclusive",
    "event": "stim_on", "event_source": "dataset_clean_fif_annotations",
    "source_sampling_rate_hz": 250.0, "output_samples": 226,
})
NOD_BASELINE = ("baseline", "none", {
    "interval_sec": [-0.1, 0.0], "interval_semantics": "half_open",
    "operation": "subtract_channel_mean",
})
NOD_EXCLUDE_EOG = ("exclude_channels", "none", {
    "exclude_names": ["HEO", "VEO"], "keep_eeg_channels": 62})
NOD_UNIT_CONVERSION = ("unit_conversion", "none", {
    "from": "V", "to": "uV", "reason": "stable_optimization_scale"})
NOD_RS200 = ("resample_poly", "none", {"target_hz": 200.0, "output_samples": 181})
NOD_RS256 = ("resample_poly", "none", {"target_hz": 256.0, "output_samples": 232})
NOD_PAD_200 = ("pad_right", "none", {
    "valid_samples": 181, "pad_samples": 19, "output_samples": 200,
    "value": 0.0, "reason": "encoder_requires_one_second_at_200hz",
})
NOD_PAD_LUNA = ("pad_right", "none", {
    "valid_samples": 232, "pad_samples": 48, "output_samples": 280,
    "value": 0.0, "reason": "pretrained_patch_size_40",
})
NOD_CLASS_SPLIT = ("class_split", "none", {
    "n_classes": 1000, "n_train_classes": 800, "n_test_classes": 200,
    "selection": "seeded_random_without_replacement", "seed": 2000,
    "split_unit": "class_id", "class_disjoint": True,
})
NOD_AGGREGATE = ("aggregate_repetitions", "none", {
    "train": "retain_all_available_class_trials",
    "test": "mean_by_class_across_sessions",
})
NOD_MNN = ("mnn_whiten", "train", {
    "covariance": "ledoit_wolf",
    "fit_unit": "training_classes_only_per_session",
    "apply_to": "train_and_test_before_test_average",
    "output_units": "dimensionless_whitened",
})
LUNA_IMAGE_CROP = ("crop", "none", {
    "interval_sec": [0.0, 1.0], "interval_semantics": "half_open",
    "n_samples": 256,
})
LUNA_ZSCORE = ("per_channel_zscore", "none", {
    "fit_unit": "individual_sample_before_padding", "std_correction": 1,
    "epsilon": 1e-8,
})
LUNA_PAD = ("pad_right", "none", {
    "valid_samples": 256, "pad_samples": 24, "output_samples": 280,
    "value": 0.0, "reason": "pretrained_patch_size_40",
})

# Grootswagers2019 (OpenNeuro ds004018): BrainVision raw at 1000 Hz, 63 EEG
# channels, Cz online reference, RSVP streams of the same 200 object images.
# The benchmark applies the paper's own filter (Hamming-windowed FIR, 0.1-100 Hz)
# and then an exact rational resample to 250 Hz; no re-referencing, no artifact
# rejection and no baseline correction, matching the paper.  The epoch is the
# paper's [-100, 1000] ms around every non-target stimulus onset; encoders with a
# fixed one-second contract consume the post-stimulus [0, 1.0] s, which resamples
# exactly onto 200 samples at 200 Hz (LaBraM/REVE) and 256 at 256 Hz (LUNA).
GROOTSWAGERS_READ = ("read_brainvision_raw", "none", {
    "source": "ds004018/sub-XX/eeg/sub-XX_task-rsvp_run-0N_eeg.vhdr",
    "events": "matching *_events.tsv (onset column in samples)",
    "sampling_rate_hz": 1000.0,
    "dataset_provided_preprocessing": [
        "Cz online reference (kept, the paper does not re-reference)",
    ],
})
GROOTSWAGERS_FILTER = ("bandpass", "none", {
    "band_hz": [0.1, 100.0], "design": "hamming_windowed_fir",
    "reason": "Grootswagers2019 NeuroImage paper filter",
})
GROOTSWAGERS_RS250 = ("resample_poly", "none", {"source_hz": 1000.0, "target_hz": 250.0})
GROOTSWAGERS_EPOCH = ("epoch", "none", {
    "interval_sec": [-0.1, 1.0], "interval_semantics": "mne_tmin_tmax_inclusive",
    "event": "stim_on_non_target", "output_samples": 276,
    "reason": "Grootswagers2019 paper epoch for non-target stimuli",
})
GROOTSWAGERS_STIMULUS_SPLIT = ("stimulus_split", "none", {
    "n_stimuli": 200, "n_train_stimuli": 180, "n_test_stimuli": 20,
    "seed": 2000, "split_unit": "stimulus",
    "selection": "seeded_random_without_replacement",
    "n_sequences_per_speed": 40,
    "reason": "THINGS-EEG2's 1654/200 disjoint-concept ratio applied to the 200 "
              "stimuli, so the reconstruction task is unseen-stimulus",
})
GROOTSWAGERS_AGGREGATE = ("aggregate_repetitions", "none", {
    "train": "retain_all_trials_of_training_stimuli",
    "test": "mean_by_held_out_stimulus_across_all_sequences",
})
GROOTSWAGERS_MNN = ("mnn_whiten", "train", {
    "covariance": "ledoit_wolf",
    "fit_unit": "training_trials_per_speed_session",
    "apply_to": "train_and_test_before_test_average",
    "output_units": "dimensionless_whitened",
    "benchmark_addition": "the paper applies no whitening",
})
GROOTSWAGERS_UNITS = ("unit_conversion", "none", {"from": "V", "to": "uV"})

# Grootswagers2025 infant dataset (OpenNeuro ds005106): EEGLAB raw at 500 Hz,
# 32 EEG channels, 5 Hz RSVP streams of the same 200 object images as the adult
# release.  Two bands are registered so retrieval can compare them:
#   * paper: the paper's own pipeline - 0.5 Hz high-pass + 40 Hz Hamming FIR,
#     average reference, epochs [-0.1, 0.8] s
#   * wide : 0.1 Hz + 100 Hz Hamming FIR plus a 50 Hz notch, because widening the
#     band re-admits the line noise that the paper's 40 Hz low-pass removes
# Both variants keep average referencing, reproducible bad-channel interpolation
# and reproducible bad-segment rejection, and both split by block (seed 2000).
G2025_READ = ("read_eeglab_raw", "none", {
    "source": "ds005106/sub-XXX/eeg/sub-XXX_task-fix_eeg.set",
    "source_format": "EEGLAB_set_continuous",
    "sampling_rate_hz": 500.0,
    "dataset_provided_preprocessing": [],
})
G2025_BAD_CHANNELS = ("bad_channel_interpolation", "none", {
    "detector": "flat_line_and_robust_log_variance_outlier",
    "threshold_z": 4.0, "interpolation": "spherical_spline",
    "reason": "the paper's clean_rawdata removes ~6.5 channels per infant",
})
G2025_FILTER_PAPER = ("bandpass", "none", {
    "band_hz": [0.5, 40.0], "design": "hamming_windowed_fir",
    "reason": "Grootswagers2025 paper filter (0.5 Hz clean_rawdata high-pass + 40 Hz FIR)",
})
G2025_FILTER_WIDE = ("bandpass", "none", {
    "band_hz": [0.1, 100.0], "design": "hamming_windowed_fir",
    "reason": "user-requested wide band; differs from the paper",
})
G2025_NOTCH = ("notch", "none", {
    "frequencies_hz": [50.0], "design": "hamming_windowed_fir",
    "reason": "the wide band re-admits 50 Hz line noise that the paper's 40 Hz low-pass removed",
})
G2025_REFERENCE = ("reference", "none", {"scheme": "average"})
G2025_BAD_SEGMENTS = ("bad_segment_rejection", "none", {
    "rule": "peak_to_peak > median + 5 * 1.4826 * MAD", "scope": "any_channel_per_epoch",
    "reason": "the paper's clean_rawdata removes ~1.5 % of segments",
})
G2025_RS250 = ("resample_poly", "none", {"source_hz": 500.0, "target_hz": 250.0})
G2025_EPOCH = ("epoch", "none", {
    "interval_sec": [-0.1, 0.8], "interval_semantics": "mne_tmin_tmax_inclusive",
    "event": "stim_on", "output_samples": 226,
    "reason": "Grootswagers2025 paper epoch",
})
# NOTE: a stimulus-disjoint 180/20 variant (unseen-stimulus, same 20 held-out
# images as the adult ds004018 release) was implemented and piloted but is NOT
# active; the parked implementation -- including a v2 of this registry -- lives
# under ``_ds005106_v2_pilot/src/``.
G2025_BLOCK_SPLIT = ("block_split", "none", {
    "split_unit": "block", "test_fraction": 0.25,
    "test_blocks_must_be_complete": True, "min_complete_blocks": 2,
    "partial_blocks": "training_only", "seed": 2000,
})
G2025_AGGREGATE = ("aggregate_repetitions", "none", {
    "train": "retain_all_train_block_trials",
    "test": "mean_by_stimulus_across_test_blocks",
})
G2025_MNN = ("mnn_whiten", "train", {
    "covariance": "ledoit_wolf", "fit_unit": "training_trials",
    "apply_to": "train_and_test_before_test_average",
    "output_units": "dimensionless_whitened",
    "benchmark_addition": "the paper applies no whitening",
})
G2025_UNITS = ("unit_conversion", "none", {"from": "V", "to": "uV"})
G2025_CROP_POST = ("crop", "none", {
    "interval_sec": [0.0, 1.0], "interval_semantics": "half_open", "n_samples": 200,
    "reason": "one-second encoder contract; only 0.8 s post-stimulus exist",
})

GROOTSWAGERS_CROP_POST = ("crop", "none", {
    "interval_sec": [0.0, 1.0], "interval_semantics": "half_open", "n_samples": 250,
    "reason": "one-second encoder contract after removal of the pre-stimulus 100 ms",
})
GROOTSWAGERS_RS200 = ("resample_poly", "none", {
    "source_hz": 250.0, "target_hz": 200.0, "output_samples": 200})
GROOTSWAGERS_RS256 = ("resample_poly", "none", {
    "source_hz": 250.0, "target_hz": 256.0, "output_samples": 256})


PROFILES = {
    "thingseeg-image-nice-v1": _p("thingseeg-image-nice-v1", "eeg_to_image",
        ("read_osf_numpy", "none", {}), RS250, IMAGE_EPOCH, MNN),
    "thingseeg-image-atm-v1": _p("thingseeg-image-atm-v1", "eeg_to_image",
        ("read_osf_numpy_raw", "none", {
            "allowed_basenames": ["raw_eeg_training.npy", "raw_eeg_test.npy"],
            "derived_inputs_forbidden": True,
            "source_filtering": "dataset_provided_filtered_raw",
        }), ATM_IMAGE_EPOCH, ATM_BASELINE, RS250, ATM_IMAGE_CROP,
        ("select_repetitions", "none", {
            "seed": 2000,
            "train_per_image_condition_per_session": 2,
            "test_per_concept_per_session": 20,
        }), MNN,
        ("aggregate_repetitions", "none", {
            "train": "retain_selected_trials",
            "test": "mean_by_concept_across_sessions",
        })),
    "thingseeg-image-reve-nice-v1": _p(
        "thingseeg-image-reve-nice-v1", "eeg_to_image",
        ("read_osf_numpy", "none", {"derived_inputs_forbidden": True}),
        RS200, IMAGE_EPOCH, MNN),
    "thingseeg-image-reve-atm-v1": _p(
        "thingseeg-image-reve-atm-v1", "eeg_to_image",
        ("read_osf_numpy_raw", "none", {
            "allowed_basenames": ["raw_eeg_training.npy", "raw_eeg_test.npy"],
            "derived_inputs_forbidden": True,
            "source_filtering": "dataset_provided_filtered_raw",
        }), ATM_IMAGE_EPOCH, ATM_BASELINE, RS200, REVE_IMAGE_CROP,
        ("select_repetitions", "none", {
            "seed": 2000,
            "train_per_image_condition_per_session": 2,
            "test_per_concept_per_session": 20,
        }), MNN,
        ("aggregate_repetitions", "none", {
            "train": "retain_selected_trials",
            "test": "mean_by_concept_across_sessions",
        })),
    "thingseeg-image-labram-atm-v1": _p(
        "thingseeg-image-labram-atm-v1", "eeg_to_image",
        ("read_osf_numpy_raw", "none", {
            "allowed_basenames": ["raw_eeg_training.npy", "raw_eeg_test.npy"],
            "derived_inputs_forbidden": True,
            "source_filtering": "dataset_provided_filtered_raw",
        }), ATM_IMAGE_EPOCH, ATM_BASELINE, RS200, LABRAM_IMAGE_CROP,
        ("select_repetitions", "none", {
            "seed": 2000,
            "train_per_image_condition_per_session": 2,
            "test_per_concept_per_session": 20,
        }), MNN,
        ("aggregate_repetitions", "none", {
            "train": "retain_selected_trials",
            "test": "mean_by_concept_across_sessions",
        })),
    "thingseeg-image-luna-nice-v1": _p(
        "thingseeg-image-luna-nice-v1", "eeg_to_image",
        ("read_osf_numpy_raw", "none", {
            "allowed_basenames": ["raw_eeg_training.npy", "raw_eeg_test.npy"],
            "derived_inputs_forbidden": True,
            "source_filtering": "dataset_provided_filtered_raw",
        }), RS256, IMAGE_EPOCH, MNN, LUNA_ZSCORE, LUNA_PAD),
    "thingseeg-image-luna-atm-v1": _p(
        "thingseeg-image-luna-atm-v1", "eeg_to_image",
        ("read_osf_numpy_raw", "none", {
            "allowed_basenames": ["raw_eeg_training.npy", "raw_eeg_test.npy"],
            "derived_inputs_forbidden": True,
            "source_filtering": "dataset_provided_filtered_raw",
        }), ATM_IMAGE_EPOCH, ATM_BASELINE, RS256, LUNA_IMAGE_CROP,
        ("select_repetitions", "none", {
            "seed": 2000,
            "train_per_image_condition_per_session": 2,
            "test_per_concept_per_session": 20,
        }), MNN,
        ("aggregate_repetitions", "none", {
            "train": "retain_selected_trials",
            "test": "mean_by_concept_across_sessions",
        }), LUNA_ZSCORE, LUNA_PAD),
    # NOD ATM starts from native BIDS EEGLAB files. The detailed-events CSV
    # is metadata only: signal timing comes from each matching events.tsv.
    "nod-image-nice-v1": _p("nod-image-nice-v1", "eeg_to_image",
        ("read_eeglab", "none", {}), RS250, IMAGE_EPOCH, MNN),
    "nod-image-nice-paper-v1": _p(
        "nod-image-nice-paper-v1", "eeg_to_image",
        ("read_openneuro_preprocessed_epochs", "none", {
            "source": "NOD-EEG derivatives/preprocessed/epochs",
            "format": "MNE-EpochsFIF",
            "dataset_preprocessing": "NOD-paper-six-step-pipeline",
        }),
        ("downsample", "none", {
            "target_hz": 250.0,
            "already_applied_by_dataset": True,
        }),
        ("bad_sensor_interpolation", "none", {
            "detector": "RANSAC",
            "interpolation": "spherical_splines",
            "already_applied_by_dataset": True,
        }),
        ("zapline", "none", {
            "line_frequency_hz": 50.0,
            "already_applied_by_dataset": True,
        }),
        ("average_reference", "none", {
            "reference": "mean_of_all_EEG_channels",
            "already_applied_by_dataset": True,
        }),
        ("ica_artifact_rejection", "none", {
            "fit_band_hz": [1.0, 100.0],
            "apply_band_hz": [0.1, 100.0],
            "component_selection": ["MNE-ICALabel", "manual_inspection"],
            "already_applied_by_dataset": True,
        }),
        ("epoch", "none", {
            "interval_sec": [-0.1, 0.8],
            "interval_semantics": "MNE_tmin_tmax_inclusive",
            "event": "stim_on",
            "already_applied_by_dataset": True,
        }),
        ("baseline", "none", {
            "interval_sec": [-0.1, 0.0],
            "operation": "subtract_channel_mean",
            "already_applied_by_dataset": True,
        }),
        ("class_split", "none", {
            "n_classes": 1000,
            "n_train_classes": 800,
            "n_test_classes": 200,
            "selection": "seeded_random_without_replacement",
            "seed": 42,
            "split_unit": "class_id",
            "class_disjoint": True,
        }),
        ("aggregate_repetitions", "none", {
            "train": "retain_all_available_class_trials",
            "test": "mean_by_class_across_sessions",
        }),
        ("mnn_whiten", "train", {
            "covariance": "ledoit_wolf",
            "fit_unit": "training_classes_only_per_session",
            "apply_to": "train_and_test",
        }),
        ("unit_conversion", "none", {
            "from": "V",
            "to": "uV",
            "reason": "stable_NICE_optimization_scale",
        }),
        version="2.0.0"),
    "nod-image-atm-v1": _p(
        "nod-image-atm-v1", "eeg_to_image",
        NOD_READ_CLEAN, NOD_EXCLUDE_EOG, NOD_EPOCH, NOD_BASELINE,
        NOD_UNIT_CONVERSION, NOD_CLASS_SPLIT, NOD_AGGREGATE, NOD_MNN,
        version="3.0.0"),
    "nod-image-reve-atm-v1": _p(
        "nod-image-reve-atm-v1", "eeg_to_image",
        NOD_READ_CLEAN, NOD_EXCLUDE_EOG, NOD_EPOCH, NOD_BASELINE,
        NOD_UNIT_CONVERSION, NOD_RS200, NOD_PAD_200,
        NOD_CLASS_SPLIT, NOD_AGGREGATE, NOD_MNN,
        version="2.0.0"),
    "nod-image-labram-atm-v1": _p(
        "nod-image-labram-atm-v1", "eeg_to_image",
        NOD_READ_CLEAN, NOD_EXCLUDE_EOG, NOD_EPOCH, NOD_BASELINE,
        NOD_UNIT_CONVERSION, NOD_RS200, NOD_PAD_200,
        NOD_CLASS_SPLIT, NOD_AGGREGATE, NOD_MNN,
        version="3.0.0"),
    "nod-image-luna-atm-v1": _p(
        "nod-image-luna-atm-v1", "eeg_to_image",
        NOD_READ_CLEAN, NOD_EXCLUDE_EOG, NOD_EPOCH, NOD_BASELINE,
        NOD_UNIT_CONVERSION, NOD_RS256, LUNA_ZSCORE, NOD_PAD_LUNA,
        NOD_CLASS_SPLIT, NOD_AGGREGATE, NOD_MNN,
        version="2.0.0"),
    # NOD retrieval shares the NOD reconstruction preprocessing exactly (same
    # clean-continuous source, epoch, baseline, class split and train-only MNN);
    # only the per-encoder sampling-rate adaptation and the task registry entry
    # differ, so retrieval consumes the same 800/200 class-disjoint rows as
    # reconstruction.
    "nod-image-nice-clean-v1": _p(
        "nod-image-nice-clean-v1", "eeg_to_image",
        NOD_READ_CLEAN, NOD_EXCLUDE_EOG, NOD_EPOCH, NOD_BASELINE,
        NOD_UNIT_CONVERSION, NOD_CLASS_SPLIT, NOD_AGGREGATE, NOD_MNN,
        version="1.0.0"),
    "nod-image-labram-nice-clean-v1": _p(
        "nod-image-labram-nice-clean-v1", "eeg_to_image",
        NOD_READ_CLEAN, NOD_EXCLUDE_EOG, NOD_EPOCH, NOD_BASELINE,
        NOD_UNIT_CONVERSION, NOD_RS200, NOD_PAD_200,
        NOD_CLASS_SPLIT, NOD_AGGREGATE, NOD_MNN,
        version="1.0.0"),
    "nod-image-reve-nice-clean-v1": _p(
        "nod-image-reve-nice-clean-v1", "eeg_to_image",
        NOD_READ_CLEAN, NOD_EXCLUDE_EOG, NOD_EPOCH, NOD_BASELINE,
        NOD_UNIT_CONVERSION, NOD_RS200, NOD_PAD_200,
        NOD_CLASS_SPLIT, NOD_AGGREGATE, NOD_MNN,
        version="1.0.0"),
    "nod-image-luna-nice-clean-v1": _p(
        "nod-image-luna-nice-clean-v1", "eeg_to_image",
        NOD_READ_CLEAN, NOD_EXCLUDE_EOG, NOD_EPOCH, NOD_BASELINE,
        NOD_UNIT_CONVERSION, NOD_RS256, LUNA_ZSCORE, NOD_PAD_LUNA,
        NOD_CLASS_SPLIT, NOD_AGGREGATE, NOD_MNN,
        version="1.0.0"),

    # ---- Grootswagers2019 (ds004018) --------------------------------------
    "grootswagers-image-nice-v1": _p(
        "grootswagers-image-nice-v1", "eeg_to_image",
        GROOTSWAGERS_READ, GROOTSWAGERS_FILTER, GROOTSWAGERS_RS250,
        GROOTSWAGERS_EPOCH, GROOTSWAGERS_STIMULUS_SPLIT,
        GROOTSWAGERS_AGGREGATE, GROOTSWAGERS_MNN, GROOTSWAGERS_UNITS,
        version="2.0.0"),
    "grootswagers-image-labram-nice-v1": _p(
        "grootswagers-image-labram-nice-v1", "eeg_to_image",
        GROOTSWAGERS_READ, GROOTSWAGERS_FILTER, GROOTSWAGERS_RS250,
        GROOTSWAGERS_EPOCH, GROOTSWAGERS_CROP_POST, GROOTSWAGERS_RS200,
        GROOTSWAGERS_STIMULUS_SPLIT, GROOTSWAGERS_AGGREGATE, GROOTSWAGERS_MNN,
        GROOTSWAGERS_UNITS, version="2.0.0"),
    "grootswagers-image-reve-nice-v1": _p(
        "grootswagers-image-reve-nice-v1", "eeg_to_image",
        GROOTSWAGERS_READ, GROOTSWAGERS_FILTER, GROOTSWAGERS_RS250,
        GROOTSWAGERS_EPOCH, GROOTSWAGERS_CROP_POST, GROOTSWAGERS_RS200,
        GROOTSWAGERS_STIMULUS_SPLIT, GROOTSWAGERS_AGGREGATE, GROOTSWAGERS_MNN,
        GROOTSWAGERS_UNITS, version="2.0.0"),
    "grootswagers-image-luna-nice-v1": _p(
        "grootswagers-image-luna-nice-v1", "eeg_to_image",
        GROOTSWAGERS_READ, GROOTSWAGERS_FILTER, GROOTSWAGERS_RS250,
        GROOTSWAGERS_EPOCH, GROOTSWAGERS_CROP_POST, GROOTSWAGERS_RS256,
        LUNA_ZSCORE, LUNA_PAD, GROOTSWAGERS_STIMULUS_SPLIT,
        GROOTSWAGERS_AGGREGATE, GROOTSWAGERS_MNN, GROOTSWAGERS_UNITS,
        version="2.0.0"),
    "grootswagers-image-atm-v1": _p(
        "grootswagers-image-atm-v1", "eeg_to_image",
        GROOTSWAGERS_READ, GROOTSWAGERS_FILTER, GROOTSWAGERS_RS250,
        GROOTSWAGERS_EPOCH, GROOTSWAGERS_STIMULUS_SPLIT,
        GROOTSWAGERS_AGGREGATE, GROOTSWAGERS_MNN, GROOTSWAGERS_UNITS,
        version="2.0.0"),
    "grootswagers-image-labram-atm-v1": _p(
        "grootswagers-image-labram-atm-v1", "eeg_to_image",
        GROOTSWAGERS_READ, GROOTSWAGERS_FILTER, GROOTSWAGERS_RS250,
        GROOTSWAGERS_EPOCH, GROOTSWAGERS_CROP_POST, GROOTSWAGERS_RS200,
        GROOTSWAGERS_STIMULUS_SPLIT, GROOTSWAGERS_AGGREGATE, GROOTSWAGERS_MNN,
        GROOTSWAGERS_UNITS, version="2.0.0"),
    "grootswagers-image-reve-atm-v1": _p(
        "grootswagers-image-reve-atm-v1", "eeg_to_image",
        GROOTSWAGERS_READ, GROOTSWAGERS_FILTER, GROOTSWAGERS_RS250,
        GROOTSWAGERS_EPOCH, GROOTSWAGERS_CROP_POST, GROOTSWAGERS_RS200,
        GROOTSWAGERS_STIMULUS_SPLIT, GROOTSWAGERS_AGGREGATE, GROOTSWAGERS_MNN,
        GROOTSWAGERS_UNITS, version="2.0.0"),
    "grootswagers-image-luna-atm-v1": _p(
        "grootswagers-image-luna-atm-v1", "eeg_to_image",
        GROOTSWAGERS_READ, GROOTSWAGERS_FILTER, GROOTSWAGERS_RS250,
        GROOTSWAGERS_EPOCH, GROOTSWAGERS_CROP_POST, GROOTSWAGERS_RS256,
        LUNA_ZSCORE, LUNA_PAD, GROOTSWAGERS_STIMULUS_SPLIT,
        GROOTSWAGERS_AGGREGATE, GROOTSWAGERS_MNN, GROOTSWAGERS_UNITS,
        version="2.0.0"),
    "grootswagers-image-classification-v1": _p(
        "grootswagers-image-classification-v1", "eeg_to_image",
        GROOTSWAGERS_READ, GROOTSWAGERS_FILTER, GROOTSWAGERS_RS250,
        GROOTSWAGERS_EPOCH, GROOTSWAGERS_STIMULUS_SPLIT,
        GROOTSWAGERS_AGGREGATE, GROOTSWAGERS_MNN, GROOTSWAGERS_UNITS,
        version="2.0.0"),
    "continuous-speech-text-v1": _p("continuous-speech-text-v1", "sentence_generation",
        READ_BV, RS250, ("align_stimulus", "none", {"source": "textgrid"}),
        ("epoch", "none", {"level": "sentence"})),
    "continuous-speech-word-v1": _p("continuous-speech-word-v1", "word_decoding",
        READ_BV, RS250, ("align_stimulus", "none", {"source": "textgrid"}),
        ("epoch", "none", {"level": "word", "duration_sec": 0.8})),
    "roamm-text-v1": _p("roamm-text-v1", "sentence_generation", READ_BV, RS250,
        ("align_stimulus", "none", {"source": "eye_tracking_fixation"}),
        ("epoch", "none", {"level": "sentence"})),
    "roamm-word-v1": _p("roamm-word-v1", "word_decoding", READ_BV, RS250,
        ("align_stimulus", "none", {"source": "eye_tracking_fixation"}),
        ("epoch", "none", {"level": "word", "duration_sec": 0.8})),
    "zuco-derived-features-v1": _p("zuco-derived-features-v1", "sentence_generation",
        ("read_matlab_features", "none", {"representation": "frequency_bands"}),
        ("per_word_zscore", "none", {"fit_unit": "individual_word"}),
        ("pad_or_truncate", "none", {"max_words": 56})),
    "speech-brain-reconstruction-v1": _p("speech-brain-reconstruction-v1", "speech_reconstruction",
        ("read_edf", "none", {"units": "uV"}), RS250,
        ("resample_audio_poly", "none", {"target_hz": 16000.0}),
        ("align_stimulus", "none", {"source": "events_tsv"})),
    "karaone-detection-v1": _p("karaone-detection-v1", "speech_detection",
        ("read_cnt", "none", {"units": "uV"}), RS250,
        ("epoch", "none", {"source": "speaking_inds"})),
    "brennan-eegnet-track-v1": _p("brennan-eegnet-track-v1", "speech_retrieval",
        ("read_fieldtrip", "none", {"units": "dataset_native"}),
        ("filter", "none", {"band_hz": [0.1, 75.0], "notch_hz": [50.0, 60.0]}),
        ("resample", "none", {"target_hz": 120.0}), ("epoch", "none", {"window_sec": [-0.5, 2.5]}),
        ("baseline", "none", {"interval_sec": [0.0, 0.2]}),
        ("robust_scale", "train", {"axis": "channel"}), ("clamp", "none", {"absolute": 20.0})),
    "brennan-labram-track-v1": _p("brennan-labram-track-v1", "speech_retrieval",
        ("read_fieldtrip", "none", {"units": "dataset_native"}),
        ("filter", "none", {"band_hz": [0.1, 75.0], "notch_hz": [50.0]}),
        ("resample", "none", {"target_hz": 200.0}), ("scale", "none", {"factor": 10000.0}),
        ("epoch", "none", {"window_sec": [-0.5, 2.5]})),
}



# Grootswagers2025 (ds005106) retrieval profiles: one pair per registered band,
# with the per-encoder sampling-rate adaptation used by the other datasets.
def _g2025_steps(band, encoder):
    filter_steps = ((G2025_FILTER_PAPER,) if band == "paper"
                    else (G2025_FILTER_WIDE, G2025_NOTCH))
    common = (G2025_READ, G2025_BAD_CHANNELS) + filter_steps + (G2025_REFERENCE,)
    tail = (G2025_BAD_SEGMENTS, G2025_RS250, G2025_EPOCH)
    if encoder == "nice":
        adapt = ()
    elif encoder == "luna":
        adapt = (G2025_CROP_POST, RS256, LUNA_ZSCORE, LUNA_PAD)
    else:
        adapt = (G2025_CROP_POST, RS200)
    return common + tail + adapt + (G2025_BLOCK_SPLIT, G2025_AGGREGATE,
                                    G2025_MNN, G2025_UNITS)


for _band in ("paper", "wide"):
    for _encoder in ("nice", "labram", "reve", "luna"):
        _pid = (f"grootswagers2025-image-nice-{_band}-v1" if _encoder == "nice"
                else f"grootswagers2025-image-{_encoder}-nice-{_band}-v1")
        PROFILES[_pid] = _p(_pid, "eeg_to_image", *_g2025_steps(_band, _encoder),
                            version="1.0.0")

# Image generation shares the same preprocessing; the encoding geometry differs
# per encoder exactly as for NOD and ds004018.
_G2025_GEN_COMMON = ((G2025_READ, G2025_BAD_CHANNELS, G2025_FILTER_PAPER,
                      G2025_REFERENCE, G2025_BAD_SEGMENTS, G2025_RS250,
                      G2025_EPOCH))
_G2025_GEN_TAIL = (G2025_BLOCK_SPLIT, G2025_AGGREGATE, G2025_MNN, G2025_UNITS)
for _encoder, _adapt in (("atm", ()), ("labram", (G2025_CROP_POST, RS200)),
                         ("reve", (G2025_CROP_POST, RS200)),
                         ("luna", (G2025_CROP_POST, RS256, LUNA_ZSCORE, LUNA_PAD))):
    _pid = (f"grootswagers2025-image-atm-paper-v1" if _encoder == "atm"
            else f"grootswagers2025-image-{_encoder}-atm-paper-v1")
    PROFILES[_pid] = _p(_pid, "eeg_to_image",
                        *(_G2025_GEN_COMMON + _adapt + _G2025_GEN_TAIL),
                        version="1.0.0")

# Concept classification: event_label carries the levelC category.
PROFILES["grootswagers2025-image-classification-paper-v1"] = _p(
    "grootswagers2025-image-classification-paper-v1", "eeg_to_image",
    *(_G2025_GEN_COMMON + _G2025_GEN_TAIL), version="1.0.0")


# 视频任务的 profile 单独放在 profiles_video.py：那一份是从视频仓库整体迁入的，
# 里面还带着图像与语音的历史定义（与 v3 的 profiles.py 重复）。这里只取
# task_id 为 video_generation 的条目，既保持 v3 的单一查询入口，又避免复制
# 两份图像 profile。
from .profiles_video import PROFILES as _VIDEO_PROFILES  # noqa: E402

PROFILES.update({
    key: value for key, value in _VIDEO_PROFILES.items()
    if value.task_id == "video_generation"
})


def get_preprocessing_profile(profile_id):
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown preprocessing profile {profile_id!r}") from exc

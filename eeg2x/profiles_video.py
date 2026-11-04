"""Versioned, model-independent preprocessing profiles."""
from __future__ import annotations

from eeg2x.protocol import PreprocessingProfile, ProcessingStep


def _p(profile_id, task_id, *steps):
    return PreprocessingProfile(
        profile_id=profile_id, task_id=task_id,
        steps=tuple(ProcessingStep(name=n, fit_scope=s, parameters=v)
                    for n, s, v in steps),
    )


READ_BV = ("read_brainvision", "none", {"type": "eeg", "units": "uV"})
RS250 = ("resample_poly", "none", {"target_hz": 250.0})
IMAGE_EPOCH = ("epoch", "none", {"t_pre_sec": 0.0, "t_post_sec": 1.0})
MNN = ("mnn_whiten", "train", {
    "covariance": "ledoit_wolf", "output_units": "dimensionless_whitened"})

PROFILES = {
    "thingseeg-image-nice-v1": _p("thingseeg-image-nice-v1", "eeg_to_image",
        ("read_osf_numpy", "none", {}), RS250, IMAGE_EPOCH, MNN),
    "nod-image-nice-v1": _p("nod-image-nice-v1", "eeg_to_image",
        ("read_eeglab", "none", {}), RS250, IMAGE_EPOCH, MNN),
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
    # SEED-DV (EEG2Video, NeurIPS 2024).  Source clock is already 200 Hz, so no
    # resampling step exists -- the profile declares the clock instead.  The
    # stored EEG values are the dataset's native scale (about 1e-4), which is
    # neither microvolt nor dimensionless, so per DATA_PROTOCOL rule 5 the unit
    # is dataset_native and must not be guessed.  Standardisation stays in the
    # model (it fits on the train split and is saved in the checkpoint), so no
    # step here carries fitted state.
    "seed-dv-video-v1": _p("seed-dv-video-v1", "video_generation",
        ("read_seed_dv_numpy", "none",
         {"units": "dataset_native", "layout": "blocks_x_channels_x_samples",
          "n_blocks": 7, "clip_seconds": 2.0}),
        ("align_stimulus", "none",
         {"source": "block_concept_repeat_manifest", "fps": 3, "n_frames": 6}),
        ("epoch", "none",
         {"clip_seconds": 2.0, "n_windows": 7, "window_samples": 100,
          "stride_samples": 50})),
}


# ===================================================================
# SEED-DV video task: per-encoder preprocessing branches
# ===================================================================
#
# The ``video_generation`` task needs one branch per encoder, because an
# encoder's input geometry is fixed by its pretraining patch size and cannot be
# imposed from outside:
#
#   eegnet  7 x 100 samples @200 Hz  the EEG2Video paper's own layout, unchanged
#   labram  3 x 200 samples @200 Hz  LaBraM's patch is 200 samples at 200 Hz
#   reve    3 x 200 samples @200 Hz  REVE's patch is 200 samples at 200 Hz
#   luna    3 x 240 samples @256 Hz  LUNA's patch is 40 samples at 256 Hz
#
# LaBraM and REVE physically cannot consume the paper's 100-sample window:
# REVE tokenises with a bare ``unfold(size=200)`` and LaBraM reshapes to
# ``(B, C, A, 200)``, so both would raise rather than underperform.
#
# The stored EEG differs per branch only where the encoder's published recipe
# requires it.  Everything else is declared and left to the model:
#
#   eegnet  raw dataset units, standardisation fitted in the model on train
#   labram  scaled volts -> microvolts (the released W->uV/100 convention);
#           the model must NOT standardise again or the scaling is cancelled
#   reve    raw dataset units; model standardises, then clamps at +/-15 as the
#           released REVE recipe does
#   luna    resampled to LUNA's 256 Hz training clock; the per-channel z-score
#           LUNA applies is stateless and lives inside the encoder
#
# This is the minimum adaptation that lets each encoder run as published.  It
# is deliberately *not* a re-tuning of the task: the clip, the split, the CLIP
# targets, the optimiser and the evaluation are identical for all four.

#: Window geometry per encoder, mirrored by
#: ``model/eeg2video_core/scheme_d_encoders.ENCODER_WINDOW_SPECS`` and written
#: into every SEED-DV HDF5 as ``encoder_window_specs_json``.
SEED_DV_ENCODER_WINDOWS = {
    "eegnet": {"window_samples": 100, "stride_samples": 50, "n_windows": 7},
    "labram": {"window_samples": 200, "stride_samples": 100, "n_windows": 3},
    "reve": {"window_samples": 200, "stride_samples": 100, "n_windows": 3},
    "luna": {"window_samples": 240, "stride_samples": 120, "n_windows": 3},
    # EEGVid's encoder (arXiv:2505.21385) is a NICE derivative whose temporal
    # branch expects a long trace, not short patches: the paper feeds it the
    # whole 2 s segment (200 Hz -> 400 samples) as one window, matching NICE's
    # own 250-sample input.  So one window spanning the epoch.
    "eegvid_nice": {"window_samples": 400, "stride_samples": 400, "n_windows": 1},
}

#: Encoders whose branch hands the model a ready-to-use representation.
SEED_DV_PREPROCESSED_ENCODERS = ("labram", "luna")

#: Encoders that carry their own published input clamp, applied in the model.
SEED_DV_ENCODER_CLIP = {"reve": 15.0}

#: Encoder -> preprocessing profile.
SEED_DV_ENCODER_PROFILES = {
    "eegnet": "seed-dv-video-v1",
    "labram": "seed-dv-video-labram-v1",
    "reve": "seed-dv-video-reve-v1",
    "luna": "seed-dv-video-luna-v1",
    "eegvid_nice": "seed-dv-video-eegvid-nice-v1",
}

#: Encoder -> native clock of the branch (LUNA is the only one that moves).
SEED_DV_ENCODER_RATE = {
    "eegnet": 200.0, "labram": 200.0, "reve": 200.0, "luna": 256.0,
    # EEGVid's encoder keeps the dataset's own clock, as the paper's
    # preprocessing does (200 Hz, 2 s).
    "eegvid_nice": 200.0,
}

#: Encoder -> stored ``eeg_units``.
SEED_DV_ENCODER_UNITS = {
    "eegnet": "dataset_native",
    "labram": "microvolt",
    "reve": "dataset_native",
    "luna": "dataset_native",
    "eegvid_nice": "dataset_native",
}

#: Encoder -> ``model_input_normalization`` recorded in the HDF5.
SEED_DV_ENCODER_MODEL_NORM = {
    encoder: ("none" if encoder in SEED_DV_PREPROCESSED_ENCODERS
              else "zscore_train_split")
    for encoder in SEED_DV_ENCODER_WINDOWS
}

_LABRAM_SCALE = 10000.0  # volts -> microvolts, then /100 as LaBraM trains
_LUNA_TARGET_HZ = 256.0


def _seed_dv_window(*, encoder):
    """The ``epoch`` step of a branch, keyed exactly as ``seed-dv-video-v1``.

    ``window_seconds`` is deliberately *not* added: the original EEGNet
    profile's parameters are ``clip_seconds``/``n_windows``/``window_samples``/
    ``stride_samples``, and the profile fingerprint written into every HDF5 is
    a hash of these dicts.  Keeping the key set identical means the EEGNet
    branch's fingerprint is unchanged by this extension.
    """

    spec = SEED_DV_ENCODER_WINDOWS[encoder]
    return ("epoch", "none", {
        "clip_seconds": 2.0,
        "n_windows": spec["n_windows"],
        "window_samples": spec["window_samples"],
        "stride_samples": spec["stride_samples"],
    })


PROFILES["seed-dv-video-labram-v1"] = _p("seed-dv-video-labram-v1", "video_generation",
    ("read_seed_dv_numpy", "none",
     {"units": "microvolt", "layout": "blocks_x_channels_x_samples",
      "n_blocks": 7, "clip_seconds": 2.0}),
    # The released LaBraM pipeline feeds EEG in microvolts divided by 100; the
    # source stores volts, so this one factor is the whole unit adaptation.
    ("scale", "none", {"factor": _LABRAM_SCALE, "from": "volts", "to": "microvolts",
                       "note": "LaBraM consumes uV/100; model standardisation disabled"}),
    ("align_stimulus", "none",
     {"source": "block_concept_repeat_manifest", "fps": 3, "n_frames": 6}),
    _seed_dv_window(encoder="labram"))

PROFILES["seed-dv-video-reve-v1"] = _p("seed-dv-video-reve-v1", "video_generation",
    ("read_seed_dv_numpy", "none",
     {"units": "dataset_native", "layout": "blocks_x_channels_x_samples",
      "n_blocks": 7, "clip_seconds": 2.0}),
    ("align_stimulus", "none",
     {"source": "block_concept_repeat_manifest", "fps": 3, "n_frames": 6}),
    _seed_dv_window(encoder="reve"),
    # Declared rather than applied here: the standardisation is Scheme D's
    # train-split transform (fitted in the model, stored in the checkpoint) and
    # the clamp is applied inside the REVE encoder wrapper.
    ("normalize_and_clip", "none",
     {"method": "zscore", "fit": "train_split", "implemented_in": "model",
      "clip": SEED_DV_ENCODER_CLIP["reve"]}))

PROFILES["seed-dv-video-luna-v1"] = _p("seed-dv-video-luna-v1", "video_generation",
    ("read_seed_dv_numpy", "none",
     {"units": "dataset_native", "layout": "blocks_x_channels_x_samples",
      "n_blocks": 7, "clip_seconds": 2.0}),
    # LUNA is the one encoder whose published clock differs from SEED-DV's.
    ("resample", "none", {"target_hz": _LUNA_TARGET_HZ, "source_hz": 200.0,
                          "method": "resample_poly"}),
    ("align_stimulus", "none",
     {"source": "block_concept_repeat_manifest", "fps": 3, "n_frames": 6}),
    _seed_dv_window(encoder="luna"),
    ("per_channel_zscore", "none",
     {"axis": "time", "eps": 1e-8, "implemented_in": "model",
      "note": "LUNA's only input normalisation; stateless, so not a fitted step"}))


PROFILES["seed-dv-video-eegvid-nice-v1"] = _p("seed-dv-video-eegvid-nice-v1", "video_generation",
    ("read_seed_dv_numpy", "none",
     {"units": "dataset_native", "layout": "blocks_x_channels_x_samples",
      "n_blocks": 7, "clip_seconds": 2.0}),
    ("align_stimulus", "none",
     {"source": "block_concept_repeat_manifest", "fps": 3, "n_frames": 6}),
    _seed_dv_window(encoder="eegvid_nice"),
    # Same as the other encoders: a train-split z-score fitted in the model and
    # stored in its checkpoint, not a fitted step of the profile.
    ("normalize_and_clip", "none",
     {"method": "zscore", "fit": "train_split", "implemented_in": "model"}))


def get_seed_dv_encoder_profile(encoder):
    """Profile id for a SEED-DV video encoder, or a KeyError naming the options."""

    try:
        return SEED_DV_ENCODER_PROFILES[encoder]
    except KeyError as exc:
        raise KeyError(
            f"Unknown SEED-DV video encoder {encoder!r}; expected one of "
            f"{sorted(SEED_DV_ENCODER_PROFILES)}"
        ) from exc


def seed_dv_window_spec(encoder):
    """The branch's window geometry, read back out of the profile it declares.

    The loader writes this into the HDF5 so the model consumes the same numbers
    the profile promised, instead of each side keeping its own copy.
    """

    profile = get_preprocessing_profile(get_seed_dv_encoder_profile(encoder))
    for step in profile.steps:
        if step.name == "epoch":
            return {
                "window_samples": int(step.parameters["window_samples"]),
                "stride_samples": int(step.parameters["stride_samples"]),
                "n_windows": int(step.parameters["n_windows"]),
            }
    raise KeyError(f"profile {profile.profile_id!r} declares no 'epoch' step")


# ===================================================================
# ds005662 video task: per-encoder preprocessing branches
# ===================================================================
#
# Source: ``/datas/eeg_benchmark_project/ds005662-download`` (OpenNeuro
# ds005662 v1.0.3, Smit et al., Sci Data 2026).  Unlike SEED-DV -- which ships
# 2 s clips on a 200 Hz clock -- this dataset ships **already epoched**
# derivatives (``derivatives/mne/sub-XX_mne_epo.fif``): 200 Hz, 64 BioSemi
# channels, 180 samples covering -0.1 to 0.795 s.  Two consequences shape the
# branch:
#
#   * The trial is shorter than one LaBraM/REVE patch (200 samples) and is not
#     a whole number of LUNA patches, so the stored epochs are right-zero-padded
#     to 200 (and LUNA's clock, 256 Hz, needs 240).  The padding is declared
#     here and written into the HDF5 so the model cannot silently disagree.
#   * The trials are epoched, so a 0.1 Hz high-pass can no longer be applied to
#     continuous data.  The band-pass is nevertheless declared and executed
#     because the protocol asks for it; the loader measures and records how
#     much it moves the samples, and the 0.1 Hz corner is warned about, because
#     its ~1.6 s time constant exceeds the 0.9 s epoch.
#
# Trial selection (``--variant base``, ``--test-block 7``):
#
#   2880 non-target trials per subject = 8 blocks x 90 stimuli x 4 flip
#   variants, exactly one trial per combination.  The branch keeps the 90
#   stimuli and the unflipped ``base`` variant only -- the flip variants are
#   the same video mirrored, and a BLIP caption cannot express orientation, so
#   including them would add test clips that are unanswerable in principle.
#   Block 7 is the held-out test set (90 clips), mirroring SEED-DV's block 7.

#: Window geometry per encoder for ds005662, after the pad to 200 samples.
DS005662_ENCODER_WINDOWS = {
    "eegnet": {"window_samples": 100, "stride_samples": 100, "n_windows": 2},
    "labram": {"window_samples": 200, "stride_samples": 200, "n_windows": 1},
    "reve": {"window_samples": 200, "stride_samples": 200, "n_windows": 1},
    "luna": {"window_samples": 240, "stride_samples": 240, "n_windows": 1},
    # Same reasoning as SEED-DV: EEGVid's encoder wants the whole epoch.  The
    # paper's preprocessing is 2 s at 200 Hz, but ds005662's stimuli are 600 ms,
    # so the epoch is the dataset's own 200 samples rather than a padded 400.
    "eegvid_nice": {"window_samples": 200, "stride_samples": 200, "n_windows": 1},
}

DS005662_ENCODER_PROFILES = {
    "eegnet": "ds005662-video-eegnet-v1",
    "labram": "ds005662-video-labram-v1",
    "reve": "ds005662-video-reve-v1",
    "luna": "ds005662-video-luna-v1",
    "eegvid_nice": "ds005662-video-eegvid-nice-v1",
}

#: Padded epoch length per encoder, *after* any resampling.
#:
#: LUNA is the odd one out: 180 samples at 200 Hz resample to 231 at 256 Hz
#: (ceil(180 * 32/25)), which is not a whole number of its 40-sample patches,
#: so it pads on to 240 = 6 patches.  The other three pad 180 -> 200, which is
#: one whole LaBraM/REVE patch and two whole EEGNet windows.
DS005662_PADDED_SAMPLES = {
    "eegnet": 200, "labram": 200, "reve": 200, "luna": 240,
    "eegvid_nice": 200,
}

DS005662_ENCODER_RATE = {
    "eegnet": 200.0, "labram": 200.0, "reve": 200.0, "luna": 256.0,
    "eegvid_nice": 200.0,
}

#: MNE returns EEG in SI units, and the released derivatives are no exception:
#: measured RMS is ~1.0e-5, i.e. volts (~10 uV), not microvolts.
DS005662_ENCODER_UNITS = {
    "eegnet": "volt", "labram": "volt", "reve": "volt", "luna": "volt",
    "eegvid_nice": "volt",
}

#: LaBraM consumes microvolts divided by 100 (its released finetune pipeline).
#: The stored data is volts, so the whole unit adaptation is one factor of
#: 1e4 -- the same convention the SEED-DV LaBraM branch uses.
DS005662_LABRAM_SCALE = 10000.0

#: EEGNet and REVE standardise in the model (train-split z-score); LaBraM and
#: LUNA receive a representation the branch already prepared.
DS005662_ENCODER_MODEL_NORM = {
    encoder: ("none" if encoder in ("labram", "luna") else "zscore_train_split")
    for encoder in DS005662_ENCODER_WINDOWS
}

DS005662_ENCODER_CLIP = {"reve": 15.0}

#: The dataset's own band-pass, as stated in the paper and recorded by MNE in
#: the derivatives' ``info`` (highpass=0.1, lowpass=100).
DS005662_BAND_HZ = (0.1, 100.0)
DS005662_NATIVE_SAMPLES = 180
DS005662_NATIVE_RATE = 200.0
DS005662_SOURCE_CHANNELS = 64
#: The paper's frame rule for the 600 ms / 25 fps stimulus clips.
DS005662_N_FRAMES = 6
DS005662_FRAME_HEIGHT = 288
DS005662_FRAME_WIDTH = 512


def _ds005662_window(*, encoder):
    spec = DS005662_ENCODER_WINDOWS[encoder]
    return ("epoch", "none", {
        "clip_seconds": round(DS005662_PADDED_SAMPLES[encoder]
                              / DS005662_ENCODER_RATE[encoder], 6),
        "n_windows": spec["n_windows"],
        "window_samples": spec["window_samples"],
        "stride_samples": spec["stride_samples"],
    })


def _ds005662_profile(encoder):
    target = DS005662_ENCODER_RATE[encoder]
    steps = [
        ("read_mne_epochs", "none", {
            "units": "microvolt", "layout": "epochs_x_channels_x_samples",
            "native_samples": DS005662_NATIVE_SAMPLES,
            "native_rate_hz": DS005662_NATIVE_RATE,
        }),
        # Executed on the released epochs, as the protocol requires.  The
        # loader records the RMS change this produces and warns about the
        # 0.1 Hz corner, whose time constant is longer than the trial.
        ("filter", "none", {
            "band_hz": list(DS005662_BAND_HZ),
            "applied_to": "released_epochs",
            "already_applied_by_source": True,
        }),
    ]
    if target != DS005662_NATIVE_RATE:
        steps.append(("resample", "none", {
            "target_hz": target, "source_hz": DS005662_NATIVE_RATE,
            "method": "resample_poly",
            "note": "applied before the pad so the patch length is exact",
        }))
    if encoder == "labram":
        steps.append(("scale", "none", {
            "factor": DS005662_LABRAM_SCALE, "from": "volts", "to": "microvolts",
            "note": "LaBraM consumes uV/100; model standardisation disabled",
        }))
    steps.append(("pad", "none", {
        "to_samples": DS005662_PADDED_SAMPLES[encoder],
        "where": "right", "value": 0.0,
        "reason": f"epoch is {DS005662_NATIVE_SAMPLES} samples; the encoder "
                  f"needs whole patches",
    }))
    steps.append(("align_stimulus", "none", {
        "source": "events_tsv_eventnumber", "variant": "base",
        "n_frames": DS005662_N_FRAMES,
    }))
    steps.append(_ds005662_window(encoder=encoder))
    if encoder == "reve":
        steps.append(("normalize_and_clip", "none", {
            "method": "zscore", "fit": "train_split", "implemented_in": "model",
            "clip": DS005662_ENCODER_CLIP["reve"],
        }))
    elif encoder == "luna":
        steps.append(("per_channel_zscore", "none", {
            "axis": "time", "eps": 1e-8, "implemented_in": "model",
        }))
    return _p(DS005662_ENCODER_PROFILES[encoder], "video_generation", *steps)


for _encoder in DS005662_ENCODER_WINDOWS:
    PROFILES[DS005662_ENCODER_PROFILES[_encoder]] = _ds005662_profile(_encoder)


def get_ds005662_encoder_profile(encoder):
    try:
        return DS005662_ENCODER_PROFILES[encoder]
    except KeyError as exc:
        raise KeyError(
            f"Unknown ds005662 encoder {encoder!r}; expected one of "
            f"{sorted(DS005662_ENCODER_PROFILES)}"
        ) from exc


# ===================================================================
# EEG-Video-Action video task: per-encoder preprocessing branches
# ===================================================================
#
# Source: ``/datas/eeg_benchmark_project/EEG-Video-Action`` -- the Single-Shot
# release of Yao, Stebner, Tuytelaars, Geirnaert & Bertrand, "Identifying
# temporal correlations between natural single-shot videos and EEG signals",
# J. Neural Eng. 21 (2024) 016018 (Zenodo record 10512414).
#
# What the release is
# -------------------
# 19 subjects x 13 single-shot videos (63 min per subject), EEGLAB ``.set``
# + ``.fdt`` per (subject, stimulus), 2048 Hz, **68 channels = 64 EEG
# (A1-A32, B1-B32) + 4 EOG (EXG1-EXG4)**.  The videos are not redistributed
# (copyright); the record publishes a YouTube link plus start/end seconds per
# stimulus instead, and those two numbers are the alignment.  Local ``.mp4``
# files are those sources, and the played clip is ``[start, end]`` inside them
# -- see ``eeg_video_action_stimuli.json`` and
# ``scripts/build_eeg_video_action_stimuli.py``, which parses the record and
# checks every row against the EEG segment lengths.
#
# Three decisions worth stating, because each one is a choice:
#
#   * **The 4 EOG channels are dropped, not regressed.**  The paper's own
#     preprocessing (section 2.2) interpolates bad channels and regresses the
#     EOG channels out of the EEG.  The benchmark does neither: SEED-DV and
#     ds005662 both arrive already cleaned from their publishers and the
#     benchmark adds no fitted cleaning of its own, so this branch keeps that
#     behaviour.  EOG regression is a fitted transform over a continuous
#     recording, which DATA_PROTOCOL forbids from ``split=test`` anyway
#     (``fit_scope=source``).  The eye-movement component therefore stays in
#     the EEG, and the per-subject bad-channel lists shipped in the release's
#     ``README.txt`` are recorded as metadata rather than interpolated.
#
#   * **0.1-100 Hz band-pass, not the paper's 0.5-15 Hz.**  The paper's
#     passband is what a 30 Hz analysis clock leaves after anti-aliasing; the
#     encoders need 200 Hz, and ds005662 declares the same 0.1-100 Hz band.
#     At 200 Hz the 100 Hz corner is exactly Nyquist, so the low-pass is a
#     no-op (ds005662's loader records the same thing) and the branch is in
#     effect a 0.1 Hz high-pass plus the decimation's own anti-aliasing.
#
#   * **2 s epochs.**  Identical to SEED-DV's clip: 400 samples at 200 Hz.
#     Every encoder branch below is therefore the same geometry SEED-DV uses,
#     declared separately here so the two can diverge later without one
#     silently dragging the other.
#
# Split: whole stimuli are held out, as in both other video datasets.  One
# video from each of the release's two recordings (``p1.bdf`` / ``p2.bdf``,
# the paper's 36 min and 35 min trials) is the test set; the other eleven are
# training.  Two different action categories are held out (dance, magic) so
# the test set is not a single scene, and every category still appears in
# training.

#: Window geometry per encoder.  The stored clip is 2 s at the same clocks as
#: SEED-DV, so these numbers are deliberately the same as
#: ``SEED_DV_ENCODER_WINDOWS``; the profiles are separate so that a future
#: change to one dataset cannot move the other.  A test pins the two together.
EEG_VIDEO_ACTION_ENCODER_WINDOWS = {
    "eegnet": {"window_samples": 100, "stride_samples": 50, "n_windows": 7},
    "labram": {"window_samples": 200, "stride_samples": 100, "n_windows": 3},
    "reve": {"window_samples": 200, "stride_samples": 100, "n_windows": 3},
    "luna": {"window_samples": 240, "stride_samples": 120, "n_windows": 3},
    "eegvid_nice": {"window_samples": 400, "stride_samples": 400, "n_windows": 1},
}

EEG_VIDEO_ACTION_PREPROCESSED_ENCODERS = ("labram", "luna")

EEG_VIDEO_ACTION_ENCODER_PROFILES = {
    "eegnet": "eeg-video-action-video-eegnet-v1",
    "labram": "eeg-video-action-video-labram-v1",
    "reve": "eeg-video-action-video-reve-v1",
    "luna": "eeg-video-action-video-luna-v1",
    "eegvid_nice": "eeg-video-action-video-eegvid-nice-v1",
}

EEG_VIDEO_ACTION_ENCODER_RATE = {
    "eegnet": 200.0, "labram": 200.0, "reve": 200.0, "luna": 256.0,
    "eegvid_nice": 200.0,
}

#: MNE's EEGLAB reader returns volts, and this release is a DC-coupled
#: BioSemi recording: raw RMS is ~1.5e-2 (dominated by drift and the very
#: strong mains the release's README warns about), while the 1-40 Hz content
#: is ~1.0e-5, i.e. 10 uV -- volts, as the montage implies.  Measured, not
#: assumed; see the loader's ``read_eeg_video_action_subject``.
EEG_VIDEO_ACTION_ENCODER_UNITS = {
    encoder: ("microvolt" if encoder == "labram" else "volt")
    for encoder in EEG_VIDEO_ACTION_ENCODER_WINDOWS
}

EEG_VIDEO_ACTION_ENCODER_MODEL_NORM = {
    encoder: ("none" if encoder in EEG_VIDEO_ACTION_PREPROCESSED_ENCODERS
              else "zscore_train_split")
    for encoder in EEG_VIDEO_ACTION_ENCODER_WINDOWS
}

EEG_VIDEO_ACTION_ENCODER_CLIP = {"reve": 15.0}

EEG_VIDEO_ACTION_LABRAM_SCALE = 10000.0

EEG_VIDEO_ACTION_NATIVE_RATE = 2048.0
EEG_VIDEO_ACTION_TARGET_RATE = 200.0
EEG_VIDEO_ACTION_SOURCE_CHANNELS = 68
EEG_VIDEO_ACTION_EEG_CHANNELS = 64
EEG_VIDEO_ACTION_EOG_CHANNELS = 4
#: The release's README asks for a rereference and a 50 Hz notch by hand; the
#: notch sits inside the passband, so the band-pass cannot stand in for it.
EEG_VIDEO_ACTION_BAND_HZ = (0.1, 100.0)
EEG_VIDEO_ACTION_NOTCH_HZ = 50.0
EEG_VIDEO_ACTION_CLIP_SECONDS = 2.0
EEG_VIDEO_ACTION_N_FRAMES = 6
EEG_VIDEO_ACTION_FRAME_HEIGHT = 288
EEG_VIDEO_ACTION_FRAME_WIDTH = 512
#: Frames are sampled at ``k * clip / n_frames`` into each epoch -- the
#: released EEG2Video rule (SEED-DV's branch uses the same 1/3 s step), which
#: excludes the epoch's final instant rather than straddling it.
EEG_VIDEO_ACTION_FRAME_RULE = "k * clip_seconds / n_frames, endpoint excluded"

#: Held-out stimuli -- one per source recording.  p1.bdf carries 01/03/04/05/
#: 06/07 and p2.bdf carries 08/09/12/13/14/15/16 (from each .set's own
#: ``pop_select`` history); the ids and their order come from the Zenodo
#: record.
EEG_VIDEO_ACTION_TEST_VIDEOS = ("05_Dance_2", "08_Magic_2")
EEG_VIDEO_ACTION_TRAIN_VIDEOS = (
    "01_Dance_1", "03_Acrob_1", "04_Magic_1", "06_Mime_2", "07_Acrob_2",
    "09_Dance_3", "12_Magic_3", "13_Dance_4", "14_Mime_3", "15_Dance_5",
    "16_Mime_6",
)


def _eeg_video_action_window(*, encoder):
    spec = EEG_VIDEO_ACTION_ENCODER_WINDOWS[encoder]
    return ("epoch", "none", {
        "clip_seconds": EEG_VIDEO_ACTION_CLIP_SECONDS,
        "n_windows": spec["n_windows"],
        "window_samples": spec["window_samples"],
        "stride_samples": spec["stride_samples"],
    })


def _eeg_video_action_profile(encoder):
    target = EEG_VIDEO_ACTION_ENCODER_RATE[encoder]
    steps = [
        ("read_eeglab", "none", {
            "units": "volt", "layout": "channels_x_samples",
            "source_channels": EEG_VIDEO_ACTION_SOURCE_CHANNELS,
            "eeg_channels": EEG_VIDEO_ACTION_EEG_CHANNELS,
            "native_rate_hz": EEG_VIDEO_ACTION_NATIVE_RATE,
        }),
        # The release's 68 channels are 64 EEG plus 4 EOG; only the EEG is
        # kept.  Declared as a step because ``n_channels`` in the HDF5 is a
        # consequence of it.
        ("select_eeg_channels", "none", {
            "keep": "non-EXG", "dropped": EEG_VIDEO_ACTION_EOG_CHANNELS,
            "eog_regression": "none",
            "bad_channel_interpolation": "none",
            "note": "the release's README.txt lists potential bad channels per "
                    "subject; recorded in the HDF5, not interpolated",
        }),
        ("rereference", "none", {
            "scheme": "average", "applied_to": "continuous",
            "note": "the release's README asks for this by hand",
        }),
        # Zero-phase, stateless, and applied on the native clock so the
        # decimation that follows starts from a band-limited signal.
        ("filter", "none", {
            "band_hz": list(EEG_VIDEO_ACTION_BAND_HZ),
            "notch_hz": [EEG_VIDEO_ACTION_NOTCH_HZ],
            "applied_at_hz": EEG_VIDEO_ACTION_NATIVE_RATE,
            "note": "0.1-100 Hz; at the stored 200 Hz the 100 Hz corner is "
                    "Nyquist, so this is effectively a 0.1 Hz high-pass plus "
                    "the decimator's anti-aliasing (same as ds005662)",
        }),
    ]
    if target != EEG_VIDEO_ACTION_NATIVE_RATE:
        steps.append(("resample", "none", {
            "target_hz": target,
            "source_hz": EEG_VIDEO_ACTION_NATIVE_RATE,
            "method": "resample_poly",
        }))
    if encoder == "labram":
        steps.append(("scale", "none", {
            "factor": EEG_VIDEO_ACTION_LABRAM_SCALE,
            "from": "volts", "to": "microvolts",
            "note": "LaBraM consumes uV/100; model standardisation disabled",
        }))
    steps.append(("align_stimulus", "none", {
        "source": "zenodo_segment_table",
        "n_frames": EEG_VIDEO_ACTION_N_FRAMES,
        "frame_rule": EEG_VIDEO_ACTION_FRAME_RULE,
        "note": "clip = [start_sec, end_sec] of the local mp4, truncated to "
                "the EEG segment length",
    }))
    steps.append(_eeg_video_action_window(encoder=encoder))
    if encoder == "reve":
        steps.append(("normalize_and_clip", "none", {
            "method": "zscore", "fit": "train_split", "implemented_in": "model",
            "clip": EEG_VIDEO_ACTION_ENCODER_CLIP["reve"],
        }))
    elif encoder == "luna":
        steps.append(("per_channel_zscore", "none", {
            "axis": "time", "eps": 1e-8, "implemented_in": "model",
        }))
    return _p(EEG_VIDEO_ACTION_ENCODER_PROFILES[encoder], "video_generation",
              *steps)


for _encoder in EEG_VIDEO_ACTION_ENCODER_WINDOWS:
    PROFILES[EEG_VIDEO_ACTION_ENCODER_PROFILES[_encoder]] = \
        _eeg_video_action_profile(_encoder)


def get_eeg_video_action_encoder_profile(encoder):
    try:
        return EEG_VIDEO_ACTION_ENCODER_PROFILES[encoder]
    except KeyError as exc:
        raise KeyError(
            f"Unknown EEG-Video-Action encoder {encoder!r}; expected one of "
            f"{sorted(EEG_VIDEO_ACTION_ENCODER_PROFILES)}"
        ) from exc


# ===================================================================
# ds008257 video task: per-encoder preprocessing branches
# ===================================================================
#
# Source: ``/datas/eeg_benchmark_project/openneurodownload/ds008257`` -- the
# EEG Moments Dataset (Gifford, Oyarzo, Zonneveld, Sartzetaki, Groen & Cichy,
# arXiv:2608.28768).  Six participants, 128-channel cap, 1,102 three-second
# naturalistic videos.
#
# What the release already is
# ---------------------------
# Unlike every other dataset here, this one ships **epoched** EEG with a
# ``stimulus_id`` per trial, so there is no alignment to derive: trial ``i`` of
# ``sub-XX_ses-YY_preprocessed_eeg.h5`` is the response to video
# ``stimulus_id[i]``.  The publisher's own preprocessing is already applied and
# is what the branch declares rather than repeats:
#
#   * 0.1-40 Hz band-pass, ICA blink removal, Autoreject trial rejection and
#     bad-channel interpolation, baseline correction against the pre-stimulus
#     interval (all in ``mne``, per the paper's Methods)
#   * recorded at 1,000 Hz, released at 500 Hz
#   * **Fz-referenced** -- Fz is the online reference and is therefore *not*
#     among the 127 stored channels
#
# Nothing is re-applied.  Re-running the band-pass would be pointless (the data
# already carries it) and re-running the 0.1 Hz high-pass over a 2 s scored
# window would actively distort it.
#
# The scoring window
# ------------------
# A trial is a 3 s video with EEG from -0.1 s to 3.5 s.  The benchmark scores
# the **middle two seconds** (0.5 s to 2.5 s), which has one large practical
# consequence: 500 Hz x 2 s = 1000 samples, and 500 -> 200 Hz turns that into
# exactly **400 samples** -- SEED-DV's clip, so every encoder branch keeps the
# geometry it was built for and no model changes are needed.  (500 -> 256 for
# LUNA is 512 samples, also as on SEED-DV.)  Starting at 0.5 s rather than 0 s
# keeps the window centred on the moment the video depicts and away from its
# first and last frames.
#
# Split: the release's own.  Videos 1-1000 are training (six repeats each) and
# 1001-1102 are test (24 repeats each), and every session contains all 102 test
# videos.  The benchmark keeps every training trial and **one trial per test
# video** -- 102 test epochs, one per stimulus class -- because scoring all
# ~2,300 test trials would cost roughly eight hours of evaluation per track for
# no extra classes.

DS008257_DATASET_ID = "ds008257"
DS008257_NATIVE_RATE = 500.0
DS008257_TARGET_RATE = 200.0
DS008257_SOURCE_CHANNELS = 127
#: The publisher's band, as recorded in the release's own metadata ``args``.
DS008257_SOURCE_BAND_HZ = (0.1, 40.0)
DS008257_SOURCE_REFERENCE = "Fz"
#: The scored window, in seconds relative to stimulus onset, inside the 3 s
#: video.  0.5 -> 2.5 s at 500 Hz is samples 300..1300 of the released epoch.
DS008257_WINDOW_START_SEC = 0.5
DS008257_WINDOW_STOP_SEC = 2.5
DS008257_CLIP_SECONDS = DS008257_WINDOW_STOP_SEC - DS008257_WINDOW_START_SEC
DS008257_N_FRAMES = 6
DS008257_FRAME_HEIGHT = 288
DS008257_FRAME_WIDTH = 512
DS008257_STIMULUS_SECONDS = 3.0
#: The release's split boundary: ids below this are training.
DS008257_TEST_ID_MIN = 1001

#: The channels LaBraM can place, in the release's own order.
#:
#: ds008257 is the first dataset here whose montage is the **extended 10-05**
#: naming (``AFF1h``, ``FFC5h``, ``PPO9h``, ``OI1h``, ...), and LaBraM's
#: pretrained positional table is a fixed 136-slot 10-20 set that does not
#: contain 50 of EMD's 127 names.  The table cannot be extended -- the slots
#: are baked into the released weights -- so the branch keeps the 77 channels
#: LaBraM can place and declares the other 50 as dropped.  That makes this
#: branch's channel count differ from the other encoders' 127, which is a
#: deliberate, recorded cost of running LaBraM at all; ``labram_input_chans``
#: raises on an unknown name precisely so this cannot happen silently.
#:
#: A test derives this list from LaBraM's own table and asserts it matches.
DS008257_LABRAM_CHANNELS = (
    "Fp1", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7",
    "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz",
    "O2", "P4", "P8", "TP10", "CP6", "CP2", "Cz", "C4",
    "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2", "AF7",
    "AF3", "AFz", "F1", "F5", "FT7", "FC3", "FCz", "C1",
    "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz",
    "PO4", "PO8", "P6", "P2", "CPz", "CP4", "TP8", "C6",
    "C2", "FC4", "FT8", "F6", "F2", "AF4", "AF8", "F9",
    "P9", "PO9", "O9", "O10", "PO10", "P10", "TPP8h", "F10",
    "FTT9h", "TTP7h", "TPP9h", "TPP10h", "FTT10h",
)

#: Window geometry per encoder.  The stored clip is 2 s at the same clocks as
#: SEED-DV, so these are deliberately SEED-DV's numbers, declared separately so
#: either dataset can change without moving the other.  A test pins the two
#: together, and another pins them to the model's own fallback table.
DS008257_ENCODER_WINDOWS = {
    "eegnet": {"window_samples": 100, "stride_samples": 50, "n_windows": 7},
    "labram": {"window_samples": 200, "stride_samples": 100, "n_windows": 3},
    "reve": {"window_samples": 200, "stride_samples": 100, "n_windows": 3},
    "luna": {"window_samples": 240, "stride_samples": 120, "n_windows": 3},
    "eegvid_nice": {"window_samples": 400, "stride_samples": 400, "n_windows": 1},
}

DS008257_PREPROCESSED_ENCODERS = ("labram", "luna")

DS008257_ENCODER_PROFILES = {
    "eegnet": "ds008257-video-eegnet-v1",
    "labram": "ds008257-video-labram-v1",
    "reve": "ds008257-video-reve-v1",
    "luna": "ds008257-video-luna-v1",
    "eegvid_nice": "ds008257-video-eegvid-nice-v1",
}

DS008257_ENCODER_RATE = {
    "eegnet": 200.0, "labram": 200.0, "reve": 200.0, "luna": 256.0,
    "eegvid_nice": 200.0,
}

#: Measured, not assumed: the released epochs have RMS ~1.0e-5, i.e. volts.
#: The scale is the publisher's own (MNE returns SI units), and nothing here
#: rescales it except the LaBraM branch below.
DS008257_ENCODER_UNITS = {
    encoder: ("microvolt" if encoder == "labram" else "volt")
    for encoder in DS008257_ENCODER_WINDOWS
}

DS008257_ENCODER_MODEL_NORM = {
    encoder: ("none" if encoder in DS008257_PREPROCESSED_ENCODERS
              else "zscore_train_split")
    for encoder in DS008257_ENCODER_WINDOWS
}

DS008257_ENCODER_CLIP = {"reve": 15.0}
DS008257_LABRAM_SCALE = 10000.0


def _ds008257_window(*, encoder):
    spec = DS008257_ENCODER_WINDOWS[encoder]
    return ("epoch", "none", {
        "clip_seconds": DS008257_CLIP_SECONDS,
        "n_windows": spec["n_windows"],
        "window_samples": spec["window_samples"],
        "stride_samples": spec["stride_samples"],
    })


def _ds008257_profile(encoder):
    target = DS008257_ENCODER_RATE[encoder]
    steps = []
    if encoder == "labram":
        # First, because everything after it works on the kept channels.
        steps.append(("select_encoder_channels", "none", {
            "keep": "channels_LaBraM_can_place",
            "n_channels": len(DS008257_LABRAM_CHANNELS),
            "dropped": DS008257_SOURCE_CHANNELS - len(DS008257_LABRAM_CHANNELS),
            "channels": list(DS008257_LABRAM_CHANNELS),
            "reason": "LaBraM's positional table is a fixed 136-slot 10-20 set "
                      "that does not contain 50 of EMD's extended 10-05 names, "
                      "and the slots are baked into the released weights",
        }))
    steps += [
        ("read_eeg_moments_h5", "none", {
            "units": "volt", "layout": "trials_x_channels_x_times",
            "source_channels": DS008257_SOURCE_CHANNELS,
            "native_rate_hz": DS008257_NATIVE_RATE,
            "reference": DS008257_SOURCE_REFERENCE,
            "baseline_corrected": True,
            "source_band_hz": list(DS008257_SOURCE_BAND_HZ),
            "note": "the release is already epoched; no alignment step exists "
                    "for this dataset",
        }),
        # Declared, not executed.  The publisher filtered the data and the
        # release carries that filter; re-running it would be a no-op at best
        # and, for the 0.1 Hz corner over a 2 s window, a distortion.
        ("filter", "none", {
            "band_hz": list(DS008257_SOURCE_BAND_HZ),
            "already_applied_by_source": True,
            "reapplied_by_benchmark": False,
            "note": "0.1-40 Hz is the publisher's own filter (paper Methods); "
                    "the branch declares it and adds none of its own",
        }),
        ("select_window", "none", {
            "start_sec": DS008257_WINDOW_START_SEC,
            "stop_sec": DS008257_WINDOW_STOP_SEC,
            "clip_seconds": DS008257_CLIP_SECONDS,
            "relative_to": "stimulus_onset",
            "source_samples": int(DS008257_CLIP_SECONDS * DS008257_NATIVE_RATE),
            "note": "middle two seconds of a three-second stimulus",
        }),
    ]
    if target != DS008257_NATIVE_RATE:
        steps.append(("resample", "none", {
            "target_hz": target,
            "source_hz": DS008257_NATIVE_RATE,
            "method": "resample_poly",
            "note": "500 -> 200 Hz turns the 2 s window into 400 samples, the "
                    "geometry every encoder branch already uses",
        }))
    if encoder == "labram":
        steps.append(("scale", "none", {
            "factor": DS008257_LABRAM_SCALE, "from": "volts", "to": "microvolts",
            "note": "LaBraM consumes uV/100; model standardisation disabled",
        }))
    steps.append(("align_stimulus", "none", {
        "source": "released_stimulus_id",
        "n_frames": DS008257_N_FRAMES,
        "frame_rule": "round((window_start + k * clip / n_frames) * n_frames "
                      "/ stimulus_seconds)",
        "note": "the release carries a stimulus_id per trial; the keyframes are "
                "decoded from that stimulus's mp4",
    }))
    steps.append(_ds008257_window(encoder=encoder))
    if encoder == "reve":
        steps.append(("normalize_and_clip", "none", {
            "method": "zscore", "fit": "train_split", "implemented_in": "model",
            "clip": DS008257_ENCODER_CLIP["reve"],
        }))
    elif encoder == "luna":
        steps.append(("per_channel_zscore", "none", {
            "axis": "time", "eps": 1e-8, "implemented_in": "model",
        }))
    return _p(DS008257_ENCODER_PROFILES[encoder], "video_generation", *steps)


for _encoder in DS008257_ENCODER_WINDOWS:
    PROFILES[DS008257_ENCODER_PROFILES[_encoder]] = _ds008257_profile(_encoder)


def get_ds008257_encoder_profile(encoder):
    try:
        return DS008257_ENCODER_PROFILES[encoder]
    except KeyError as exc:
        raise KeyError(
            f"Unknown ds008257 encoder {encoder!r}; expected one of "
            f"{sorted(DS008257_ENCODER_PROFILES)}"
        ) from exc


def get_preprocessing_profile(profile_id):
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown preprocessing profile {profile_id!r}") from exc

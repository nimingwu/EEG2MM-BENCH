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
    # ZuCo1 raw EEG branch used by the GLIM-only-EEG reproduction.  The
    # source array is the protocol-versioned 128-Hz GLIM preprocessing output
    # (104 measured channels plus 24 explicit zero padding channels).  The
    # LaBraM branch applies the exact rational 128 -> 200 Hz polyphase
    # resampling before writing its HDF5; no fitted/test state is used.
    "zuco1-glim-raw-v1": _p("zuco1-glim-raw-v1", "sentence_generation",
        ("read_derived_glim_dataframe", "none", {
            "source_sampling_hz": 128.0, "source_channels": 128,
            "source_representation": "raw_sentence_eeg"}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "zuco1-labram-raw-v1": _p("zuco1-labram-raw-v1", "sentence_generation",
        ("read_derived_glim_dataframe", "none", {
            "source_sampling_hz": 128.0, "source_channels": 128,
            "source_representation": "raw_sentence_eeg"}),
        ("resample_poly", "none", {"target_hz": 200.0,
                                     "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "zuco1-luna-raw-v1": _p("zuco1-luna-raw-v1", "sentence_generation",
        ("read_derived_glim_dataframe", "none", {
            "source_sampling_hz": 128.0, "source_channels": 128,
            "source_representation": "raw_sentence_eeg"}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "zuco1-reve-raw-v1": _p("zuco1-reve-raw-v1", "sentence_generation",
        ("read_derived_glim_dataframe", "none", {
            "source_sampling_hz": 128.0, "source_channels": 128,
            "source_representation": "raw_sentence_eeg"}),
        ("resample_poly", "none", {"target_hz": 200.0,
                                     "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    # ZuCo2 mirrors the ZuCo1 raw-EEG branches byte for byte: same 128-Hz GLIM
    # preprocessing output, same 128-channel layout, same eager null prompt.
    # The two datasets get separate profile ids because DATA_PROTOCOL.md
    # requires dataset-specific rules to live in their own versioned profile
    # rather than being inferred from a model name.
    "zuco2-glim-raw-v1": _p("zuco2-glim-raw-v1", "sentence_generation",
        ("read_derived_glim_dataframe", "none", {
            "source_sampling_hz": 128.0, "source_channels": 128,
            "source_representation": "raw_sentence_eeg"}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "zuco2-labram-raw-v1": _p("zuco2-labram-raw-v1", "sentence_generation",
        ("read_derived_glim_dataframe", "none", {
            "source_sampling_hz": 128.0, "source_channels": 128,
            "source_representation": "raw_sentence_eeg"}),
        ("resample_poly", "none", {"target_hz": 200.0,
                                     "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "zuco2-luna-raw-v1": _p("zuco2-luna-raw-v1", "sentence_generation",
        ("read_derived_glim_dataframe", "none", {
            "source_sampling_hz": 128.0, "source_channels": 128,
            "source_representation": "raw_sentence_eeg"}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "zuco2-reve-raw-v1": _p("zuco2-reve-raw-v1", "sentence_generation",
        ("read_derived_glim_dataframe", "none", {
            "source_sampling_hz": 128.0, "source_channels": 128,
            "source_representation": "raw_sentence_eeg"}),
        ("resample_poly", "none", {"target_hz": 200.0,
                                     "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    # ------------------------------------------------------------------
    # Chisco (ds005170) — Chinese imagined speech, 5 subjects.
    #
    # Source is the dataset's OWN published preprocessing output
    # (`derivatives/preprocessed_pkl`), not the raw 1 kHz EDFs.  An earlier
    # revision of this profile re-ran PREP -> AutoReject -> ICA from the EDFs;
    # that was dropped because it cost a measured 41 min/recording (224
    # recordings ~ 6.4 days) and could not reproduce the authors' settings
    # exactly (their script only reaches sub-01..03).  Consuming their released
    # derivatives is both cheaper and strictly more faithful.
    #
    # Bandwidth: the authors apply a 1 Hz high-pass and deliberately NO low-pass
    # (verified in the fif metadata: highpass=1.0, lowpass=250.0).  A high-pass
    # cannot be undone from a derivative, so this benchmark supplies only the
    # missing side -> the effective band is 1-100 Hz, with the "1" inherited.
    #
    # fit_scope is "none" throughout: nothing here is fitted, and because the
    # upstream PREP/ICA ran per-recording inside the dataset itself, no fitted
    # artifact crosses the train/test boundary.  Sentence lists are ~98.7%
    # unique per run (6681 rows / 6567 unique over 45 runs), so a text-based
    # split is very nearly a run-based split.
    #
    # The four branches differ ONLY in output rate, mirroring ZuCo: glim/luna
    # keep 128 Hz / 1280 samples and labram/reve use 200 Hz / 2000 samples, so
    # the existing runners and encoders work unchanged.  At 128 Hz the Nyquist
    # limit is 64 Hz, so the 100 Hz low-pass is bounded by the output rate on
    # those two branches -- the accepted cost of matching ZuCo's contract.
    "chisco-glim-raw-v1": _p("chisco-glim-raw-v1", "sentence_generation",
        ("read_chisco_derivative", "none", {
            "source": "derivatives/preprocessed_pkl", "phase": "imagine",
            "source_sampling_hz": 500.0, "source_channels": 125,
            "eeg_channels": 122,
            "inherited_highpass_hz": 1.0, "inherited_lowpass_hz": 250.0,
            "note": "dataset's own PREP/AutoReject/ICA output, not re-applied"}),
        ("lowpass", "none", {"h_freq": 100.0, "method": "fir_zero_phase"}),
        ("select_eeg", "none", {"keep": "eeg_only",
                                 "drop": ["VEO", "HEO", "Trigger"]}),
        ("resample_poly", "none", {"target_hz": 128.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "chisco-luna-raw-v1": _p("chisco-luna-raw-v1", "sentence_generation",
        ("read_chisco_derivative", "none", {
            "source": "derivatives/preprocessed_pkl", "phase": "imagine",
            "source_sampling_hz": 500.0, "source_channels": 125,
            "eeg_channels": 122,
            "inherited_highpass_hz": 1.0, "inherited_lowpass_hz": 250.0,
            "note": "dataset's own PREP/AutoReject/ICA output, not re-applied"}),
        ("lowpass", "none", {"h_freq": 100.0, "method": "fir_zero_phase"}),
        ("select_eeg", "none", {"keep": "eeg_only",
                                 "drop": ["VEO", "HEO", "Trigger"]}),
        ("resample_poly", "none", {"target_hz": 128.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "chisco-labram-raw-v1": _p("chisco-labram-raw-v1", "sentence_generation",
        ("read_chisco_derivative", "none", {
            "source": "derivatives/preprocessed_pkl", "phase": "imagine",
            "source_sampling_hz": 500.0, "source_channels": 125,
            "eeg_channels": 122,
            "inherited_highpass_hz": 1.0, "inherited_lowpass_hz": 250.0,
            "note": "dataset's own PREP/AutoReject/ICA output, not re-applied"}),
        ("lowpass", "none", {"h_freq": 100.0, "method": "fir_zero_phase"}),
        ("select_eeg", "none", {"keep": "eeg_only",
                                 "drop": ["VEO", "HEO", "Trigger"]}),
        ("resample_poly", "none", {"target_hz": 200.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "chisco-reve-raw-v1": _p("chisco-reve-raw-v1", "sentence_generation",
        ("read_chisco_derivative", "none", {
            "source": "derivatives/preprocessed_pkl", "phase": "imagine",
            "source_sampling_hz": 500.0, "source_channels": 125,
            "eeg_channels": 122,
            "inherited_highpass_hz": 1.0, "inherited_lowpass_hz": 250.0,
            "note": "dataset's own PREP/AutoReject/ICA output, not re-applied"}),
        ("lowpass", "none", {"h_freq": 100.0, "method": "fir_zero_phase"}),
        ("select_eeg", "none", {"keep": "eeg_only",
                                 "drop": ["VEO", "HEO", "Trigger"]}),
        ("resample_poly", "none", {"target_hz": 200.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    # ------------------------------------------------------------------
    # Chisco-2.0 (ds006317, "COFETT") -- Chinese imagined speech, 2 subjects.
    #
    # Unlike ds005170 this dataset publishes NO derivatives: only raw 1 kHz
    # EDFs whose header reads highpass=0.0 / lowpass=500.0, i.e. completely
    # unfiltered.  The authors' paper (arXiv 2607.18749v1 §E + Figure 8)
    # describes the pipeline but no code is published, so
    # `preprocess_chisco2_seed2000.py` reimplements the DESCRIPTION:
    #
    #     500 Hz -> PREP (bad channels + robust reference)
    #            -> 50 Hz notch + 1 Hz zero-phase high-pass FIR
    #            -> Autoreject -> EOG regression (HEO/VEO)
    #            -> ICA(30, extended Infomax) + MNE-ICALabel
    #            -> the recall / inner-speech segment
    #
    # NO LOW-PASS is applied -- the paper's explicit choice, made to keep the
    # full band up to 250 Hz rather than to save a filter design.  The steps
    # below therefore contain no `lowpass`, unlike every other Chisco/ZuCo
    # branch.  The 128 Hz branches are still Nyquist-bounded to 64 Hz by
    # `resample_poly` itself, which is inherent to resampling rather than a
    # band choice.
    #
    # fit_scope is "none" for the steps below because THIS benchmark's steps
    # (read -> select -> resample -> pad -> prompt) fit nothing.  That is not a
    # claim that the data is unfitted: PREP/Autoreject/EOG/ICA all fit on the
    # recording, they run upstream in the preprocess script, and under a
    # text-level split a recording contributes rows to more than one split, so
    # those fitted artifacts do cross the train/test boundary.  DATA_PROTOCOL
    # has no honest fit_scope for that ("source" is rejected on split=test,
    # "train" would require fitting on the training rows only, "none" would be
    # false), so it is declared in `preprocessing_state["fit_scope_note"]`
    # instead -- the same convention the ds005170 branch already uses for the
    # authors' own PREP/ICA.
    #
    # The 4 branches differ ONLY in output rate, mirroring ZuCo and Chisco.
    "chisco2-glim-raw-v1": _p("chisco2-glim-raw-v1", "sentence_generation",
        ("read_chisco2_derivative", "none", {
            "source": "seed2000_source/chisco2_eeg_500hz.df",
            "phase": "imagine", "source_sampling_hz": 500.0,
            "source_channels": 122,
            "note": "PREP/Autoreject/EOG-regression/ICA applied upstream by "
                    "preprocess_chisco2_seed2000.py, not re-applied here"}),
        ("select_eeg", "none", {"keep": "eeg_only",
                                 "drop": ["VEO", "HEO", "Trigger"]}),
        ("resample_poly", "none", {"target_hz": 128.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "chisco2-luna-raw-v1": _p("chisco2-luna-raw-v1", "sentence_generation",
        ("read_chisco2_derivative", "none", {
            "source": "seed2000_source/chisco2_eeg_500hz.df",
            "phase": "imagine", "source_sampling_hz": 500.0,
            "source_channels": 122,
            "note": "PREP/Autoreject/EOG-regression/ICA applied upstream by "
                    "preprocess_chisco2_seed2000.py, not re-applied here"}),
        ("select_eeg", "none", {"keep": "eeg_only",
                                 "drop": ["VEO", "HEO", "Trigger"]}),
        ("resample_poly", "none", {"target_hz": 128.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "chisco2-labram-raw-v1": _p("chisco2-labram-raw-v1", "sentence_generation",
        ("read_chisco2_derivative", "none", {
            "source": "seed2000_source/chisco2_eeg_500hz.df",
            "phase": "imagine", "source_sampling_hz": 500.0,
            "source_channels": 122,
            "note": "PREP/Autoreject/EOG-regression/ICA applied upstream by "
                    "preprocess_chisco2_seed2000.py, not re-applied here"}),
        ("select_eeg", "none", {"keep": "eeg_only",
                                 "drop": ["VEO", "HEO", "Trigger"]}),
        ("resample_poly", "none", {"target_hz": 200.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "chisco2-reve-raw-v1": _p("chisco2-reve-raw-v1", "sentence_generation",
        ("read_chisco2_derivative", "none", {
            "source": "seed2000_source/chisco2_eeg_500hz.df",
            "phase": "imagine", "source_sampling_hz": 500.0,
            "source_channels": 122,
            "note": "PREP/Autoreject/EOG-regression/ICA applied upstream by "
                    "preprocess_chisco2_seed2000.py, not re-applied here"}),
        ("select_eeg", "none", {"keep": "eeg_only",
                                 "drop": ["VEO", "HEO", "Trigger"]}),
        ("resample_poly", "none", {"target_hz": 200.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    # ------------------------------------------------------------------
    # TMNRED (ds005383) -- Chinese natural reading, 30 subjects, 30 channels.
    #
    # Source is the raw BIDS EDFs (200 Hz, unfiltered: header says
    # highpass=0.0).  The published `derivatives/preproc/*z.set` is NOT used
    # because its epochs carry only three event codes while the raw events
    # carry 50 distinct labels -- an epoch cannot be mapped back to its
    # sentence.  See preprocess_tmnred_seed2000.py.
    #
    # Band is the dataset's own published scheme (paper s41597-025-05036-2 and
    # the released pre.m): average reference, 0.5-80 Hz FIR + 50 Hz notch,
    # ICA with pop_icflag thresholds.  Their downsample-to-200 Hz step is a
    # no-op because the BIDS EDFs are already 200 Hz, so it is recorded and not
    # re-applied.
    #
    # CHANNEL COUNT -- the one place these profiles differ structurally
    # ------------------------------------------------------------------
    # DATA_PROTOCOL.md:31 leaves the channel count to each dataset adapter, and
    # TMNRED has 30 real channels.  labram/reve/luna take those 30 unchanged:
    # LUNA and REVE derive electrode positions from `_fixed_positions(c, ...)`
    # and LaBraM picks a subset of its 128 pretrained channel position
    # embeddings through `input_chans`.  `glim` pads to 32 because GLIM's
    # EEGEncoder uses the channel count as its input blocks' hidden_dim and
    # shares num_heads with the 256-wide decoder, so num_heads must divide
    # gcd(n_channels, 256); 30 admits only {1, 2}.  That is also why Chisco's
    # 122 channels were padded to 128.  The two padding channels are named
    # PAD000/PAD001, never disguised as electrodes.
    #
    # fit_scope is "none" for the steps below because THIS benchmark's steps fit
    # nothing; the recording-level fits (bad channels, ICA, bad trials) run
    # upstream and are declared in `preprocessing_state`.  Same convention as
    # the chisco/chisco2 branches.
    "tmnred-glim-raw-v1": _p("tmnred-glim-raw-v1", "sentence_generation",
        ("read_tmnred_derivative", "none", {
            "source": "seed2000_source/tmnred_eeg_200hz.df",
            "source_sampling_hz": 200.0, "source_channels": 30,
            "n_channels_out": 32,
            "note": "bad channels + average reference + 0.5-80 Hz + ICA applied "
                    "upstream by preprocess_tmnred_seed2000.py; 30 -> 32 zero "
                    "padding because GLIM's num_heads must divide n_channels"}),
        ("select_eeg", "none", {"keep": "eeg_only"}),
        ("resample_poly", "none", {"target_hz": 128.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "tmnred-luna-raw-v1": _p("tmnred-luna-raw-v1", "sentence_generation",
        ("read_tmnred_derivative", "none", {
            "source": "seed2000_source/tmnred_eeg_200hz.df",
            "source_sampling_hz": 200.0, "source_channels": 30,
            "n_channels_out": 30,
            "note": "real 30 channels, no padding; LUNA is channel-count "
                    "agnostic via _fixed_positions(c, ...)"}),
        ("select_eeg", "none", {"keep": "eeg_only"}),
        ("resample_poly", "none", {"target_hz": 128.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 1280}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "tmnred-labram-raw-v1": _p("tmnred-labram-raw-v1", "sentence_generation",
        ("read_tmnred_derivative", "none", {
            "source": "seed2000_source/tmnred_eeg_200hz.df",
            "source_sampling_hz": 200.0, "source_channels": 30,
            "n_channels_out": 30,
            "note": "real 30 channels, no padding; LaBraM selects 30 of its 128 "
                    "pretrained position embeddings via input_chans"}),
        ("select_eeg", "none", {"keep": "eeg_only"}),
        ("resample_poly", "none", {"target_hz": 200.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
    "tmnred-reve-raw-v1": _p("tmnred-reve-raw-v1", "sentence_generation",
        ("read_tmnred_derivative", "none", {
            "source": "seed2000_source/tmnred_eeg_200hz.df",
            "source_sampling_hz": 200.0, "source_channels": 30,
            "n_channels_out": 30,
            "note": "real 30 channels, no padding; REVE is channel-count "
                    "agnostic via _fixed_positions(c, ...)"}),
        ("select_eeg", "none", {"keep": "eeg_only"}),
        ("resample_poly", "none", {"target_hz": 200.0, "anti_alias": True}),
        ("pad_or_truncate", "none", {"samples": 2000}),
        ("null_prompt", "none", {"task": True, "dataset": True,
                                   "subject": True})),
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


def get_preprocessing_profile(profile_id):
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown preprocessing profile {profile_id!r}") from exc

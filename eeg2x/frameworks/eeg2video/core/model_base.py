"""BaseModel — abstract interface for all EEG models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class BaseModel(ABC):
    """All models must inherit this and implement generate().

    Lifecycle:
        model = MyModel()
        model.load()
        outputs = model.generate(inputs)   # inputs = list of EEGSample-like dicts
        model.unload()

    Optional:
        model.train(config)  # if the model supports training
    """

    @abstractmethod
    def load(self):
        """Load model weights, initialise resources. Called once before generate()."""
        ...

    @abstractmethod
    def generate(self, inputs: List[Any]) -> List[Any]:
        """Run inference on a batch of samples.

        Parameters
        ----------
        inputs : list of dict
            Each dict has at minimum the fields produced by Sample.from_h5_row()
            or h5-based ground-truth extraction:
              - sample_id : str
              - eeg_data  : ndarray (n_channels, n_times) or data_ref string
              - ground_truth : varies by task

        Returns
        -------
        outputs : list of dict
            Each dict's structure must match what benchmark_eval.py expects
            for the corresponding task's prediction format, e.g.:
              image_retrieval:  {"ranked_indices": [0, 7, 3]}
              word_decoding:    {"predicted_word": "hello"}
              sentence_gen:     {"generated_text": "..."}
              image_generation: {"path": "/path/to/gen/image.jpg"}
        """
        ...

    def train(self, config=None):
        """Optional: train the model. Raises NotImplementedError by default.

        Override this if your model supports training (e.g. deep learning models).
        The config dict typically contains data paths, hyperparameters, etc.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement train(). "
            "Use a separate training script or override train()."
        )

    def unload(self):
        """Release GPU memory, close files. Called once after generate()."""
        pass

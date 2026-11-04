"""EEG-To-Text（Wang & Ji, AAAI 2022）词级特征 + BART baseline。"""
from .runner import run_inference

__all__ = ["run_inference"]

import logging

import torch
import whisper

logger = logging.getLogger(__name__)

_WHISPER_HOP_LENGTH = 160

def preprocess_audio(path: str, n_mels: int = 80) -> tuple[torch.Tensor, int]:
    audio = whisper.load_audio(str(path))
    n_frames = min(len(audio) // _WHISPER_HOP_LENGTH, 3000)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
    return mel, n_frames

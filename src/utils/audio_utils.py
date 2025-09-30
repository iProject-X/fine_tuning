"""
Audio processing utilities
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import io
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing utilities for Whisper"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and convert to tensor

        Args:
            audio_path: Path to audio file

        Returns:
            Audio tensor (mono, 16kHz)
        """
        try:
            # Load audio
            waveform, orig_sample_rate = torchaudio.load(audio_path)

            # Resample if needed
            if orig_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_sample_rate, self.sample_rate
                )
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            return waveform.squeeze(0)

        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise

    async def process_audio_bytes(self, audio_data: bytes) -> torch.Tensor:
        """
        Process audio from bytes

        Args:
            audio_data: Raw audio bytes

        Returns:
            Processed audio tensor
        """
        try:
            # Create temporary file-like object
            audio_io = io.BytesIO(audio_data)

            # Load from bytes
            waveform, orig_sample_rate = torchaudio.load(audio_io)

            # Resample if needed
            if orig_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_sample_rate, self.sample_rate
                )
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            return waveform.squeeze(0)

        except Exception as e:
            logger.error(f"Failed to process audio bytes: {e}")
            raise

    async def process_audio(self, audio_path: str) -> dict:
        """
        Process audio and return metadata

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio info and quality metrics
        """
        try:
            audio = self.load_audio(audio_path)

            # Calculate quality metrics
            duration = len(audio) / self.sample_rate
            rms_energy = torch.sqrt(torch.mean(audio ** 2))

            # Simple quality assessment
            is_valid = (
                duration >= 1.0 and  # At least 1 second
                duration <= 60.0 and  # At most 60 seconds
                rms_energy > 0.001  # Not silent
            )

            quality_score = min(1.0, float(rms_energy) * 10)

            return {
                'duration': float(duration),
                'sample_rate': int(self.sample_rate),
                'rms_energy': float(rms_energy),
                'quality_score': float(quality_score),
                'is_valid': bool(is_valid)
            }

        except Exception as e:
            logger.error(f"Failed to process audio {audio_path}: {e}")
            return {
                'duration': 0.0,
                'sample_rate': self.sample_rate,
                'rms_energy': 0.0,
                'quality_score': 0.0,
                'is_valid': False
            }
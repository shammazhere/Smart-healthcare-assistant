# ============================================================
# test_hybrid_sarvam.py — Tests for Sarvam AI primary + Fallback
# ============================================================
import os
import wave
import io
import unittest
from unittest.mock import patch, MagicMock

from tts_engine import (
    is_sarvam_available as is_sarvam_tts,
    resolve_sarvam_language,
    decode_wav_to_pcm16,
)
from stt_engine import (
    is_sarvam_available as is_sarvam_stt,
    pcm_to_wav_bytes,
    normalize_language_code,
)


class TestHybridSarvam(unittest.TestCase):

    def test_sarvam_detection(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": ""}, clear=True):
            self.assertFalse(is_sarvam_tts())
            self.assertFalse(is_sarvam_stt())

        with patch.dict(os.environ, {"SARVAM_API_KEY": "valid-sarvam-key-123"}):
            self.assertTrue(is_sarvam_tts())
            self.assertTrue(is_sarvam_stt())

    def test_sarvam_language_mapping(self):
        self.assertEqual(resolve_sarvam_language("hi"), "hi-IN")
        self.assertEqual(resolve_sarvam_language("hindi"), "hi-IN")
        self.assertEqual(resolve_sarvam_language("ta"), "ta-IN")
        self.assertEqual(resolve_sarvam_language("kannada"), "kn-IN")
        self.assertEqual(resolve_sarvam_language("en"), "en-IN")
        self.assertEqual(resolve_sarvam_language("unknown"), "en-IN")

    def test_pcm_to_wav_packing(self):
        dummy_pcm = b"\x00\x00" * 16000  # 1 sec of 16-bit 16kHz audio
        wav_bytes = pcm_to_wav_bytes(dummy_pcm, sample_rate=16000, channels=1)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))

        # Verify WAV header
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getnframes(), 16000)

    def test_language_normalization(self):
        self.assertEqual(normalize_language_code("hi"), "hi")
        self.assertEqual(normalize_language_code("hindi"), "hi")
        self.assertEqual(normalize_language_code("ta"), "ta")
        self.assertEqual(normalize_language_code("kannada"), "kn")
        self.assertEqual(normalize_language_code("invalid_lang"), "en")


if __name__ == "__main__":
    unittest.main()

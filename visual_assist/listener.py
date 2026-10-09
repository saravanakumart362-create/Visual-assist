"""Speech-to-text input from the microphone.

Backends:
  vosk     - offline recognition (needs a downloaded Vosk model).
  google   - Google Speech Recognition (needs internet).
  keyboard - type the command; for development without a microphone.

Every backend exposes listen() -> str | None (None = nothing heard).
"""

from __future__ import annotations

import json
import logging
import queue
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class KeyboardListener:
    def listen(self) -> Optional[str]:
        # EOFError (stdin closed) is left to the caller: it means "no more input".
        return input("command> ").strip() or None


class VoskListener:
    def __init__(self, cfg, resolve=Path):
        import sounddevice
        import vosk

        model = resolve(cfg.vosk_model)
        if not Path(model).is_dir():
            raise RuntimeError(f"Vosk model not found at {model}")
        vosk.SetLogLevel(-1)
        self._sd = sounddevice
        self._recogniser = vosk.KaldiRecognizer(vosk.Model(str(model)), SAMPLE_RATE)
        self._timeout = cfg.timeout_s

    def listen(self) -> Optional[str]:
        audio = queue.Queue()
        with self._sd.RawInputStream(
            samplerate=SAMPLE_RATE, blocksize=8000, dtype="int16", channels=1,
            callback=lambda data, frames, when, status: audio.put(bytes(data)),
        ):
            deadline = time.monotonic() + self._timeout
            while time.monotonic() < deadline:
                try:
                    chunk = audio.get(timeout=0.5)
                except queue.Empty:
                    continue
                if self._recogniser.AcceptWaveform(chunk):
                    text = json.loads(self._recogniser.Result()).get("text", "")
                    if text:
                        return text
        self._recogniser.Reset()
        return None


class GoogleListener:
    def __init__(self, cfg, resolve=Path):
        import speech_recognition as sr

        self._sr = sr
        self._recogniser = sr.Recognizer()
        self._mic = sr.Microphone()
        self._language = cfg.language
        self._timeout = cfg.timeout_s
        with self._mic as source:
            self._recogniser.adjust_for_ambient_noise(source, duration=1.0)

    def listen(self) -> Optional[str]:
        try:
            with self._mic as source:
                audio = self._recogniser.listen(
                    source, timeout=self._timeout, phrase_time_limit=6
                )
            return self._recogniser.recognize_google(audio, language=self._language)
        except (self._sr.WaitTimeoutError, self._sr.UnknownValueError):
            return None
        except self._sr.RequestError as exc:
            log.warning("Google Speech Recognition unreachable: %s", exc)
            return None


def open_listener(cfg, resolve=Path):
    backends = {"vosk": VoskListener, "google": GoogleListener}
    if cfg.backend == "none":
        return None
    if cfg.backend == "keyboard":
        return KeyboardListener()
    if cfg.backend in backends:
        return backends[cfg.backend](cfg, resolve)
    for name, backend in backends.items():
        try:
            return backend(cfg, resolve)
        except Exception as exc:
            log.info("%s listener not available (%s).", name, exc)
    log.warning("No microphone backend available; type commands instead.")
    return KeyboardListener()

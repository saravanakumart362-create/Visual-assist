"""Text-to-speech output to the earphone, in any of the supported languages.

Voices:
  offline - espeak-ng on the Pi, the built-in Windows voices on a PC.
            Always available, robotic for Indian languages.
  gtts    - Google Text-to-Speech: natural voices for English, Malayalam,
            Hindi and Tamil, but it needs an internet connection.
  console - print only (no audio device).

"auto" uses the offline voice for English and Google's for other
languages. Whatever is chosen, the offline voice stands behind it, so a
dropped connection never leaves the user in silence.

Speech runs on a worker thread so detection never waits for a sentence to
finish, and an urgent warning cuts off whatever is being said.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

RETRY_AFTER_S = 60.0  # how long a failed voice is left alone before retrying

# Windows: speak stdin with an installed voice for the language, exit 3 if none.
_SAPI = (
    "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; "
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$v = $s.GetInstalledVoices() | Where-Object {{ $_.Enabled -and "
    "$_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq '{lang}' }} | Select-Object -First 1; "
    "if ($v) {{ $s.SelectVoice($v.VoiceInfo.Name) }} elseif ('{lang}' -ne 'en') {{ exit 3 }}; "
    "$s.Rate = {rate}; $s.Speak([Console]::In.ReadToEnd())"
)
# Windows: play the audio file named in an environment variable, wait until it ends.
_AUDIO_VARIABLE = "VISUAL_ASSIST_AUDIO"
_PLAY = (
    "Add-Type -AssemblyName PresentationCore; "
    "$p = New-Object System.Windows.Media.MediaPlayer; "
    "$p.Open([uri]$env:VISUAL_ASSIST_AUDIO); $p.Play(); "
    "$n = 0; while (-not $p.NaturalDuration.HasTimeSpan -and $n -lt 100) "
    "{ Start-Sleep -Milliseconds 50; $n++ }; "
    "if ($p.NaturalDuration.HasTimeSpan) "
    "{ Start-Sleep -Milliseconds ($p.NaturalDuration.TimeSpan.TotalMilliseconds + 150) }"
)


class _Voice:
    """Runs a command per utterance; the command can be cut short."""

    def __init__(self):
        self._process = None
        self._stopped = False

    def _run(self, command: list, stdin: bytes = b"", **environment) -> None:
        self._stopped = False
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env={**os.environ, **environment})
        self._process.communicate(stdin)
        if self._process.returncode and not self._stopped:
            raise RuntimeError(f"{command[0]} exited with code {self._process.returncode}")

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            self._stopped = True
            process.kill()


class OfflineVoice(_Voice):
    def __init__(self, language: str, rate: int):
        super().__init__()
        if sys.platform == "win32":
            # SAPI rate runs from -10 to 10; 0 is roughly 160 words per minute.
            sapi_rate = max(-10, min(10, round((rate - 160) / 20)))
            self._command = ["powershell", "-NoProfile", "-Command",
                             _SAPI.format(lang=language, rate=sapi_rate)]
        else:
            espeak = shutil.which("espeak-ng") or shutil.which("espeak")
            if not espeak:
                raise RuntimeError("espeak-ng is not installed (sudo apt install espeak-ng)")
            self._command = [espeak, "-v", language, "-s", str(rate), "--stdin"]

    def speak(self, text: str) -> None:
        # Text goes in on stdin so nothing in it is interpreted by a shell.
        self._run(self._command, text.encode("utf-8"))


class GoogleVoice(_Voice):
    def __init__(self, language: str):
        super().__init__()
        from gtts import gTTS

        self._gtts = gTTS
        self._language = language
        self._cache = Path(tempfile.gettempdir()) / "visual_assist_tts"
        self._cache.mkdir(exist_ok=True)
        self._mpg123 = shutil.which("mpg123")
        if not self._mpg123 and sys.platform != "win32":
            raise RuntimeError("no MP3 player found (sudo apt install mpg123)")

    def speak(self, text: str) -> None:
        # Phrases repeat a lot ("Scanning."), so each is only fetched once.
        key = hashlib.sha1(f"{self._language}:{text}".encode("utf-8")).hexdigest()
        path = self._cache / f"{key}.mp3"
        if not path.exists():
            partial = path.with_suffix(".part")
            self._gtts(text=text, lang=self._language, timeout=6).save(str(partial))
            partial.replace(path)
        if self._mpg123:
            self._run([self._mpg123, "-q", str(path)])
        else:
            self._run(["powershell", "-NoProfile", "-Command", _PLAY],
                      **{_AUDIO_VARIABLE: str(path)})


def _make_voices(backend: str, language: str, rate: int) -> list:
    """The voices to try, in order, for a backend setting."""
    if backend == "console":
        return []
    wanted = []
    if backend == "gtts" or (backend == "auto" and language != "en"):
        wanted.append(lambda: GoogleVoice(language))
    wanted.append(lambda: OfflineVoice(language, rate))
    voices = []
    for make in wanted:
        try:
            voices.append(make())
        except Exception as exc:
            log.warning("A speech voice is not available: %s", exc)
    return voices


class Speaker:
    def __init__(self, cfg):
        self._backend = cfg.backend
        self.language = cfg.language
        self.rate = cfg.rate
        self.last = ""  # the most recent sentence, for "repeat"
        self._lock = threading.Lock()
        self._voices = _make_voices(self._backend, self.language, self.rate)
        self._failed = {}  # voice -> time it last failed
        self._current = None
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._queue.unfinished_tasks > 0

    @property
    def voices(self) -> list:
        """Names of the voices in use, first choice first."""
        return [type(voice).__name__ for voice in self._voices]

    def configure(self, language: str = None, rate: int = None) -> None:
        """Change the language or speed for everything said from now on."""
        with self._lock:
            self.language = language or self.language
            self.rate = rate or self.rate
            self._voices = _make_voices(self._backend, self.language, self.rate)
            self._failed.clear()

    def say(self, text: str, wait: bool = False, urgent: bool = False) -> None:
        """Queue a sentence. `urgent` drops what is queued and cuts in now."""
        try:
            print(f"[TTS]: {text}", flush=True)
        except UnicodeEncodeError:
            print(f"[TTS]: {text.encode('ascii', 'backslashreplace').decode()}", flush=True)
        self.last = text
        if urgent:
            self._interrupt()
        self._queue.put(text)
        if wait:
            self.wait()

    def wait(self) -> None:
        """Block until everything queued has been spoken."""
        self._queue.join()

    def close(self) -> None:
        self.wait()
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _interrupt(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
        current = self._current
        if current is not None:
            current.stop()

    def _speak(self, text: str) -> None:
        with self._lock:
            voices = list(self._voices)
        for voice in voices:
            if time.monotonic() - self._failed.get(voice, float("-inf")) < RETRY_AFTER_S:
                continue
            self._current = voice
            try:
                voice.speak(text)
                return
            except Exception as exc:
                # Usually "no internet" for Google's voice: fall through to the next.
                log.warning("%s failed (%s); trying the next voice.",
                            type(voice).__name__, exc)
                self._failed[voice] = time.monotonic()
            finally:
                self._current = None

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                self._speak(text)
            except Exception:
                log.exception("Speech output failed")
            finally:
                self._queue.task_done()

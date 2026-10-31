
import os
import time
import tempfile
import requests

from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_MODEL = os.getenv(
    "SARVAM_STT_MODEL",
    "saaras:v2.5"
)

SARVAM_URL = "https://api.sarvam.ai/speech-to-text-translate"

SARVAM_PIECE_SECONDS = 25

_model = None


# ---------------- WHISPER ----------------

def load_model():

    global _model

    if _model is None:

        import whisper

        print(
            f"Loading Whisper model: {WHISPER_MODEL} ..."
        )

        _model = whisper.load_model(WHISPER_MODEL)

        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"].strip()


# ---------------- SARVAM API ----------------

def _send_to_sarvam(piece_path: str) -> str:

    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing.")

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    data = {
        "model": SARVAM_MODEL,
        "with_diarization": "false"
    }

    with open(piece_path, "rb") as audio_file:

        files = {
            "file": (
                os.path.basename(piece_path),
                audio_file,
                "audio/wav"
            )
        }

        response = requests.post(
            SARVAM_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=(30, 180)
        )

    if not response.ok:

        print("Sarvam API Status:", response.status_code)
        print("Sarvam API Response:", response.text)

        response.raise_for_status()

    result = response.json()

    transcript = result.get("transcript")

    if not transcript:
        raise RuntimeError(
            f"Transcript missing in Sarvam response: {result}"
        )

    return transcript.strip()


def transcribe_chunk_sarvam(
    chunk_path: str,
    retries: int = 3
) -> str:

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is missing. Check your .env file."
        )

    if not os.path.exists(chunk_path):
        raise FileNotFoundError(chunk_path)

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    total_pieces = (
        len(audio) + piece_ms - 1
    ) // piece_ms

    transcripts = []

    for i, start in enumerate(range(0, len(audio), piece_ms)):

        piece = audio[start:start + piece_ms]

        # Create a temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        piece_path = temp_file.name
        temp_file.close()

        try:

            piece.export(piece_path, format="wav")

            print(
                f"  Sarvam piece {i + 1}/{total_pieces}"
            )

            for attempt in range(1, retries + 1):

                try:

                    transcript = _send_to_sarvam(piece_path)

                    transcripts.append(transcript)

                    break

                except requests.RequestException as error:

                    print(
                        f"Sarvam request failed: {error}"
                    )

                    if attempt == retries:
                        raise

                    wait_time = attempt * 5

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

        finally:

            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(transcripts)


# ---------------- ROUTING ----------------

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    language = language.lower().strip()

    if language == "hinglish":

        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    if not chunks:
        raise ValueError("No audio chunks provided.")

    language = language.lower().strip()

    engine = (
        "Sarvam AI"
        if language == "hinglish"
        else "Whisper"
    )

    print(f"Using {engine} for transcription.")

    results = []

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk {i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        results.append(text)

    print("Transcription complete.")

    return " ".join(results)

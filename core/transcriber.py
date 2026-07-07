import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# "cpu" + "int8" is the fast, low-memory default. If you have a CUDA GPU,
# set WHISPER_DEVICE=cuda and WHISPER_COMPUTE_TYPE=float16 in your .env.
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# How many chunks to transcribe/translate in parallel. Whisper chunks are
# CPU-bound (raise cautiously, based on your core count); Sarvam chunks are
# network-bound (safe to raise higher).
MAX_WORKERS_WHISPER = int(os.getenv("MAX_WORKERS_WHISPER", "2"))
MAX_WORKERS_SARVAM = int(os.getenv("MAX_WORKERS_SARVAM", "4"))

SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():

    global _model

    if _model is None:
        print(f"Loading faster-whisper model: {WHISPER_MODEL} "
              f"(device={WHISPER_DEVICE}, compute_type={WHISPER_COMPUTE_TYPE}) ...")
        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:

    model = load_model()

    segments, _info = model.transcribe(chunk_path, task="transcribe")
    return " ".join(segment.text for segment in segments).strip()


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    sarvam_api_key = os.getenv("SARVAM_API_KEY")
    headers = {"api-subscription-key": sarvam_api_key}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not os.getenv("SARVAM_API_KEY"):
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

   



def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:

    is_hinglish = language.lower() == "hinglish"
    engine = "Sarvam AI" if is_hinglish else "Whisper"
    max_workers = MAX_WORKERS_SARVAM if is_hinglish else MAX_WORKERS_WHISPER

    print(f"Using {engine} for transcription "
          f"({len(chunks)} chunk(s), {max_workers} worker(s) in parallel).")

    # Pre-load the model once, in the main thread, before spinning up workers
    # (faster-whisper's WhisperModel is safe to share across threads for
    # inference; loading it once avoids duplicate loads racing each other).
    if not is_hinglish:
        load_model()

    results = [None] * len(chunks)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(transcribe_chunk, chunk, language): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            results[i] = future.result()
            completed += 1
            print(f"Transcribed chunk {completed}/{len(chunks)} (chunk #{i + 1}).")

    full_transcript = " ".join(results)

    print("Transcription complete.")

    return full_transcript.strip()
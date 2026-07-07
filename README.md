# 🎬 AI Video Assistant

Turn any recorded video or a meeting — a YouTube link or a local audio/video file — into a searchable, summarised meeting record. The pipeline transcribes the recording, generates a title and summary, extracts action items / key decisions / open questions, and lets you chat with the transcript using RAG.

## Features

- **Flexible input** — paste a YouTube URL or point to a local audio/video file
- **Fast local transcription** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend, ~4x faster than `openai-whisper` on CPU), with chunks transcribed in parallel
- **Hinglish support** — routes Hindi/Hinglish audio to [Sarvam AI](https://www.sarvam.ai/)'s speech-to-text-translate API instead of Whisper
- **LLM-powered analysis** via [Mistral](https://mistral.ai/) (through LangChain LCEL):
  - Meeting title generation
  - Map-reduce summarisation (handles long transcripts by chunking)
  - Action item / key decision / open question extraction
- **Chat with your meeting** — a RAG chain over a local [ChromaDB](https://www.trychroma.com/) vector store (HuggingFace `all-MiniLM-L6-v2` embeddings), so you can ask follow-up questions grounded in the transcript
- **Two interfaces** — a Streamlit web UI (`app.py`) and a CLI (`main.py`)

## Tech Stack

`Python` · `Streamlit` · `LangChain` (LCEL) · `Mistral` · `faster-whisper` · `Sarvam AI` · `ChromaDB` · `HuggingFace embeddings` · `yt-dlp` · `pydub`

## Project Structure

```
ai_video_summarizer/
├── app.py                    # Streamlit UI
├── main.py                   # CLI entry point
├── test.py                   # Manual pipeline smoke test
├── requirements.txt
├── core/
│   ├── transcriber.py         # Whisper (English) + Sarvam (Hinglish) transcription
│   ├── summarize.py            # Title generation + map-reduce summarisation
│   ├── extractor.py             # Action items / decisions / questions extraction
│   ├── vector_store.py          # ChromaDB vector store + embeddings
│   └── rag_engine.py             # LCEL RAG chain for chatting with the transcript
└── utils/
    └── audio_processor.py      # YouTube download / file conversion / chunking
```

## Setup

**Requirements:** Python 3.10+, [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH.

```bash
git clone https://github.com/AashiLad/ai_video_summarizer.git
cd ai_video_summarizer
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key      # only needed for hinglish transcription
```

> **Note:** `.env` must be loaded *before* `core`/`utils` modules are imported, since they read environment variables at import time. Both `app.py` and `main.py` already call `load_dotenv()` first thing, before any other project imports — keep it that way if you add new entry points.

## Usage

**Streamlit UI:**
```bash
streamlit run app.py
```
Paste a YouTube URL or local file path in the sidebar, pick the spoken language, and hit **Analyse**.

**CLI:**
```bash
python main.py
```
You'll be prompted for a source and language, then dropped into a chat loop once analysis completes.

## Configuration

Optional environment variables (all have sensible defaults):

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `small` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large-v3`) |
| `WHISPER_DEVICE` | `cpu` | Set to `cuda` if you have a GPU |
| `WHISPER_COMPUTE_TYPE` | `int8` | Set to `float16` when using `cuda` |
| `MAX_WORKERS_WHISPER` | `2` | Parallel Whisper workers (CPU-bound, raise cautiously) |
| `MAX_WORKERS_SARVAM` | `4` | Parallel Sarvam API calls (network-bound, safe to raise) |
| `SARVAM_STT_MODEL` | `saaras:v2.5` | Sarvam speech-to-text-translate model |



## License

MIT

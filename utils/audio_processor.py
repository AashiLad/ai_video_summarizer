import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': "wav",
                'preferredquality': '192',
            }
        ],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename


# now function for files like mp3, mp4 audios: Whisper needs mono, 16kHz
def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16kHz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)  # wav path leta che
    # chunk divide in minutes but ae ms ma hoi so multiply krvanuu by 60*1000
    # eg agar 50 min nu audio che then 50*60 krvanu toh 3000 minutes made then convert this to ms so that chunking can be done
    chunk_ms = chunk_minutes * 60 * 1000
    # eg after this we get 3000000 ms which is 50 min in ms ena pachi
    # now we will create a list of chunks to store the audio segments
    chunks = []
    # loop levanu so that audio ne chunks ma divide kari sakiye by interval of chunk_ms which le 6lakh millisec
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        # 0 , 3000000 , 6lakh (10*60*1000) ave audio wav file convert thayi
        chunk = audio[start: start + chunk_ms]  # 0th ms thi 6lakh ms sudhi chunk karvanu
        chunk_path = f"{wav_path}_chunk_{i}.wav"  # chunk path banavanu
        chunk.export(chunk_path, format="wav")  # chunk export karvanu
        chunks.append(chunk_path)  # chunk path append kairu list ma
    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks

import yt_dlp
from pydub import AudioSegment
import os

import shutil

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": output_path,
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
    }],
    "quiet": True,
    "noplaylist": True,
    "retries": 5,
    "extractor_retries": 3,
    "remote_components": ["ejs:github"],
    "force_ipv4": True,
    "cachedir": False,
    "verbose": True,
}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        wav_filename = os.path.splitext(filename)[0] + ".wav"

    print(f"Audio downloaded: {wav_filename}")
    return wav_filename


def convert_to_wav(input_file: str) -> str:
    output_path = os.path.splitext(input_file)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)

    chunk_length_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start + chunk_length_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith(("http://", "https://")):
        print("Downloading audio from YouTube...")
        wav_path = download_youtube_audio(source)
    else:
        print("Converting local audio file to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)

    print(f"Created {len(chunks)} audio chunks.")
    return chunks



import shutil

print("yt-dlp:", yt_dlp.version.__version__)
print("FFmpeg:", shutil.which("ffmpeg"))
print("Deno:", shutil.which("deno"))
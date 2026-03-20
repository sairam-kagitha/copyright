import subprocess
import os

def extract_audio_from_video(video_path, output_wav="temp_audio.wav"):
    """
    Extract audio from mp4 video using ffmpeg
    """

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_wav
import subprocess
import os

def extract_audio_from_video(video_path, output_wav="extracted_audio.wav"):

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

    subprocess.run(command)

    return output_wav


video_file = "video_input/Speech_Recognition_Test_Video_Generation.mp4"

audio_file = extract_audio_from_video(video_file)

print("Audio extracted to:", audio_file)

# Verify file exists
if os.path.exists(audio_file):
    print("Extraction successful. You can play:", audio_file)
else:
    print("Extraction failed")
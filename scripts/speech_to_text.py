import whisper

print("Loading Whisper model...")

model = whisper.load_model("base")

print("Model loaded!")

audio_path = input("Enter audio file path: ")

print("Transcribing audio...")

result = model.transcribe(audio_path)

print("\nTranscript:\n")
print(result["text"])

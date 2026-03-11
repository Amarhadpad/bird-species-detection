import os
from pydub import AudioSegment

DATASET_PATH = "dataset"

print("Starting MP3 → WAV conversion...\n")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith(".mp3"):

            mp3_path = os.path.join(root, file)
            wav_path = os.path.join(root, file.replace(".mp3", ".wav"))

            try:
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format="wav")

                print("Converted:", mp3_path)

                # Optional: delete original mp3
                os.remove(mp3_path)

            except Exception as e:
                print("Failed:", mp3_path)

print("\nConversion completed.")
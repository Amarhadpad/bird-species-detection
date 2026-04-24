import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio(file_path):

    y, sr = librosa.load(file_path, sr=22050)

    plt.figure(figsize=(12, 10))

    # --------------------------------------
    # 1. Waveform
    # --------------------------------------
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")

    # --------------------------------------
    # 2. Spectrogram
    # --------------------------------------
    plt.subplot(4, 1, 2)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")

    # --------------------------------------
    # 3. MFCC
    # --------------------------------------
    plt.subplot(4, 1, 3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")

    # --------------------------------------
    # 4. Chroma
    # --------------------------------------
    plt.subplot(4, 1, 4)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title("Chroma")

    plt.tight_layout()
    plt.show()


# Run directly
if __name__ == "__main__":
    file = input("Enter audio file path: ")
    visualize_audio(file)
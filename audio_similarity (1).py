import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import time

# -------------------------------------------------
# 1. Record Audio from User
# -------------------------------------------------
def record_audio(filename, duration=5, fs=22050):
    print(f"\nRecording {filename} for {duration} seconds...")
    print("Please speak now.")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)

    for i in range(duration, 0, -1):
        print(f"Recording... {i} seconds remaining")
        time.sleep(1)

    sd.wait()
    sf.write(filename, audio, fs)
    print(f"{filename} recorded successfully.")

# Record two audio samples
record_audio("audio1.wav")
record_audio("audio2.wav")

# -------------------------------------------------
# 2. Load Recorded Audio
# -------------------------------------------------
audio1, sr = librosa.load("audio1.wav", sr=22050, mono=True)
audio2, sr = librosa.load("audio2.wav", sr=22050, mono=True)

# -------------------------------------------------
# 3. Preprocessing
# -------------------------------------------------
min_len = min(len(audio1), len(audio2))
audio1 = audio1[:min_len]
audio2 = audio2[:min_len]

audio1 = audio1 / np.max(np.abs(audio1))
audio2 = audio2 / np.max(np.abs(audio2))

# -------------------------------------------------
# 4. Frequency Domain Analysis (FFT)
# -------------------------------------------------
fft1 = np.abs(np.fft.fft(audio1))
fft2 = np.abs(np.fft.fft(audio2))

# Use only positive frequencies
fft1 = fft1[:len(fft1)//2]
fft2 = fft2[:len(fft2)//2]

# -------------------------------------------------
# 5. Similarity Calculation (CORRECT METHOD)
# -------------------------------------------------
# Normalize full frequency vectors
fft1_norm = fft1 / np.linalg.norm(fft1)
fft2_norm = fft2 / np.linalg.norm(fft2)

# Cosine similarity using frequency distribution
similarity_score = cosine_similarity(
    fft1_norm.reshape(1, -1),
    fft2_norm.reshape(1, -1)
)[0][0]

# -------------------------------------------------
# 6. Visualization (Single Graph + Text)
# -------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(fft1, label="Audio 1", alpha=0.7)
plt.plot(fft2, label="Audio 2", alpha=0.7)

plt.text(
    0.5, 0.9,
    f"Similarity Score = {round(similarity_score, 3)}",
    fontsize=12,
    ha='center',
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.title("Frequency Domain Comparison of Two Audio Signals")
plt.xlabel("Frequency Bins")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()

# Graph stays open until user closes it
plt.show()

# -------------------------------------------------
# 7. Output Result (Text)
# -------------------------------------------------
print("\nFinal Similarity Score:", round(similarity_score, 3))
if similarity_score > 0.8:
    print("The audio samples are very similar.")
elif similarity_score > 0.5:
    print("The audio samples are somewhat similar.")
else:
    print("The audio samples are different.")   

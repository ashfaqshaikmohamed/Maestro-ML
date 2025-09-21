# Count audio files per genre
genre_counts = {g: len(os.listdir(os.path.join(DATA_DIR, g))) for g in genres}
df_counts = pd.DataFrame(list(genre_counts.items()), columns=["Genre", "Count"])

sns.barplot(x="Genre", y="Count", data=df_counts)
plt.title("Number of Samples per Genre")
plt.xticks(rotation=45)
plt.show()

# Inspect one audio file
example_file = os.path.join(DATA_DIR, genres[0], os.listdir(os.path.join(DATA_DIR, genres[0]))[0])
y, sr = librosa.load(example_file, duration=30)

print(f"Audio shape: {y.shape}, Sample rate: {sr}")

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform Example")
plt.show()

# Compute spectrogram
D = np.abs(librosa.stft(y))
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram Example")
plt.show()


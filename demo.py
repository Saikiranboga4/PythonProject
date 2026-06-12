import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf
import librosa
import librosa.display

# ── 1. Fetch stock data ──────────────────────────────────────────
ticker = "NSAI"
data = yf.download(ticker, period="1d", interval="1m")
graph_values = data['Close'].values

# ── 2. Plot the stock graph ──────────────────────────────────────
plt.plot(graph_values)
plt.title(f"{ticker} — This becomes your music")
plt.xlabel("Days")
plt.ylabel("Price")
plt.pause(3)
plt.close()

# ── 3. Normalize & map to frequencies ───────────────────────────
normalized = (graph_values - graph_values.min()) / (graph_values.max() - graph_values.min())

freq_min, freq_max = 200, 1200   # tweak for mood
frequencies = freq_min + normalized * (freq_max - freq_min)

# ── 4. Synthesize audio with effects ────────────────────────────
sample_rate = 44100
duration_per_point = 0.1       # tweak for tempo

audio = []

for freq in frequencies:
    t = np.linspace(0, duration_per_point, int(sample_rate * duration_per_point), endpoint=False)

    # Harmonics — richer, less robotic
    wave = (0.5 * np.sin(2 * np.pi * freq * t) +
            0.3 * np.sin(2 * np.pi * freq * 2 * t) +
            0.2 * np.sin(2 * np.pi * freq * 3 * t))

    # Smooth fade in/out — removes harsh clicks
    envelope = np.hanning(len(wave))
    audio.append(wave * envelope)

audio_signal = np.concatenate(audio)

# Normalize final signal to prevent clipping
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# ── 5. Export ────────────────────────────────────────────────────
sf.write("output_musicN.wav", audio_signal, sample_rate)
print("Done — output_music.wav generated")

D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_signal)), ref=np.max)

plt.figure(figsize=(14, 6))

# Top — stock price graph
plt.subplot(2, 1, 1)
plt.plot(graph_values, color='cyan')
plt.title(f"{ticker} — Stock Price")
plt.xlabel("Days")
plt.ylabel("Price ($)")

# Bottom — spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title(f"{ticker} — Audio Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()
plt.savefig("spectrogram.png")
plt.show()
print("Spectrogram saved — spectrogram.png")

# ── 8. Animated spectrogram video ───────────────────────────────
# print("Generating video — this may take a minute...")

# fps = 24
# total_duration = len(audio_signal) / sample_rate
# total_frames = int(total_duration * fps)

# fig_anim, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
# fig_anim.patch.set_facecolor('black')

# # Top — stock price (static, always visible)
# ax1.plot(graph_values, color='cyan', linewidth=1.5)
# ax1.set_facecolor('black')
# ax1.set_title(f"{ticker} — Stock Price", color='white')
# ax1.tick_params(colors='white')
# ax1.set_xlabel("Days", color='white')
# ax1.set_ylabel("Price ($)", color='white')

# # Bottom — spectrogram base
# librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', cmap='magma', ax=ax2)
# ax2.set_facecolor('black')
# ax2.set_title(f"{ticker} — Spectrogram", color='white')
# ax2.tick_params(colors='white')
# ax2.set_xlabel("Time (s)", color='white')
# ax2.set_ylabel("Frequency (Hz)", color='white')

# # Playhead line — moves across spectrogram as video plays
# playhead = ax2.axvline(x=0, color='white', linewidth=1.5)

# plt.tight_layout()

# def update(frame):
#     current_time = frame / fps
#     playhead.set_xdata([current_time, current_time])
#     return playhead,

# ani = animation.FuncAnimation(
#     fig_anim,
#     update,
#     frames=total_frames,
#     interval=1000 / fps,
#     blit=True
# )

# # Export as mp4
# writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
# ani.save("spectrogram_video.mp4", writer=writer)
# plt.close()
# print("Video saved — spectrogram_video.mp4")
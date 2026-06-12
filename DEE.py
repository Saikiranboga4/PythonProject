import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf
import librosa
import librosa.display
import cv2

# ── 1. Fetch stock data ──────────────────────────────────────────
ticker = "AAPL"
data = yf.download(ticker, period="1d", interval="1m")
graph_values = data['Close'].values

# ── 2. Plot the stock graph ──────────────────────────────────────
plt.plot(graph_values, color='cyan')
plt.title(f"{ticker} — This becomes your music")
plt.xlabel("Days")
plt.ylabel("Price")
plt.pause(3)
plt.close()

# ── 3. Normalize & map to frequencies ───────────────────────────
normalized = (graph_values - graph_values.min()) / (graph_values.max() - graph_values.min())
freq_min, freq_max = 200, 1200
frequencies = freq_min + normalized * (freq_max - freq_min)

# ── 4. Synthesize audio with effects ────────────────────────────
sample_rate = 44100
duration_per_point = 0.2

audio = []
for freq in frequencies:
    t = np.linspace(0, duration_per_point, int(sample_rate * duration_per_point), endpoint=False)
    wave = (0.5 * np.sin(2 * np.pi * freq * t) +
            0.3 * np.sin(2 * np.pi * freq * 2 * t) +
            0.2 * np.sin(2 * np.pi * freq * 3 * t))
    envelope = np.hanning(len(wave))
    audio.append(wave * envelope)

audio_signal = np.concatenate(audio)
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# ── 5. Export audio ──────────────────────────────────────────────
sf.write("output_music.wav", audio_signal, sample_rate)
print("Done — output_music.wav generated")

# ── 6. Build neon bar visualizer video ──────────────────────────
print("Generating neon bar visualizer video...")

fps         = 30
width       = 1280
height      = 720
n_bars      = 64
chunk_size  = int(sample_rate / fps)

# Dark blue background — BGR for OpenCV
bg_color    = (40, 10, 0)        # dark blue in BGR

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter("visualizer_video.mp4", fourcc, fps, (width, height))

total_chunks = len(audio_signal) // chunk_size

for i in range(total_chunks):
    chunk = audio_signal[i * chunk_size:(i + 1) * chunk_size]

    # FFT — get frequency magnitudes
    fft_vals  = np.abs(np.fft.rfft(chunk, n=chunk_size))
    fft_vals  = fft_vals[:n_bars]
    fft_vals  = fft_vals / (np.max(fft_vals) + 1e-6)   # normalize 0–1

    # Draw frame
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

    bar_width  = width // n_bars
    max_height = int(height * 0.85)

    for b in range(n_bars):
        bar_h = int(fft_vals[b] * max_height)
        x1    = b * bar_width + 2
        x2    = x1 + bar_width - 4
        y1    = height - bar_h
        y2    = height

        # Neon gradient — blue → cyan → pink based on bar position
        ratio = b / n_bars
        if ratio < 0.5:
            # Blue to cyan
            r = int(20  + ratio * 2 * 0)
            g = int(100 + ratio * 2 * 155)
            b_col = int(255)
        else:
            # Cyan to pink
            r = int(0   + (ratio - 0.5) * 2 * 255)
            g = int(255 - (ratio - 0.5) * 2 * 155)
            b_col = int(255 - (ratio - 0.5) * 2 * 100)

        color_bgr = (b_col, g, r)

        # Main bar
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, -1)

        # Glow effect — slightly wider, transparent overlay
        glow_color = (min(b_col + 40, 255), min(g + 40, 255), min(r + 40, 255))
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2), glow_color, 1)

        # Top cap highlight
        cv2.rectangle(frame, (x1, y1 - 4), (x2, y1), (255, 255, 255), -1)

    # Ticker label
    cv2.putText(frame, f"{ticker} — Stock Music Visualizer",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (200, 200, 255), 2, cv2.LINE_AA)

    # Progress bar at bottom
    progress = int((i / total_chunks) * width)
    cv2.rectangle(frame, (0, height - 6), (progress, height), (255, 180, 80), -1)

    out.write(frame)

out.release()
print("Video saved — visualizer_video.mp4")
print("Tip: Open with VLC for best playback")
import subprocess

# ── 7. Merge video + audio using FFmpeg ─────────────────────────
print("Merging audio and video...")

subprocess.run([
    "ffmpeg", "-y",
    "-i", "visualizer_video.mp4",   # video input
    "-i", "output_music.wav",        # audio input
    "-c:v", "copy",                  # keep video as is
    "-c:a", "aac",                   # encode audio to aac
    "-shortest",                     # stop at shortest stream
    "final_output.mp4"               # final merged output
])

print("Done — final_output.mp4 generated")
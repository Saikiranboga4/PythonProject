import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal

# ── 1. Fetch stock data ──────────────────────────────────────────
ticker = "^NSEI"
data = yf.download(ticker, period="1d", interval="1m")
if data.empty:
    print(f"No data found for {ticker} — check the ticker symbol")
    exit()
graph_values = data['Close'].values.flatten()

# ── 2. Plot the stock graph ──────────────────────────────────────
plt.plot(graph_values)
plt.title(f"{ticker} — This becomes your music")
plt.xlabel("Days")
plt.ylabel("Price")
# plt.savefig(f"{ticker}_stock_graph.png")
plt.pause(0.01)
plt.close()

# ── 3. Normalize & map to frequencies ───────────────────────────
normalized = (graph_values - graph_values.min()) / (graph_values.max() - graph_values.min())

freq_min, freq_max = 200, 1200
frequencies = freq_min + normalized * (freq_max - freq_min)

# ── 4. Synthesize audio ──────────────────────────────────────────
sample_rate = 44100
duration_per_point = 0.2

audio = []

for freq in frequencies:
    t = np.linspace(0, duration_per_point, int(sample_rate * duration_per_point), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    audio.append(wave)

audio_signal = np.concatenate(audio)

# # ── 5. Export ────────────────────────────────────────────────────


# ── SKRILLEX EDM EFFECTS ─────────────────────────────────────────

# 1. WOBBLE BASS — LFO modulates frequency over time (dubstep growl)
print("Applying wobble bass...")
lfo_rate    = 4.0        # wobble speed in Hz — higher = faster wobble
lfo_depth   = 0.6        # how deep the wobble goes 0–1
t_full      = np.linspace(0, len(audio_signal) / sample_rate, len(audio_signal))
lfo         = 1.0 - lfo_depth * (0.5 + 0.5 * np.sin(2 * np.pi * lfo_rate * t_full))
audio_signal = audio_signal * lfo

# 2. HARD DISTORTION — clip and saturate like a synth saw
print("Applying distortion...")
drive       = 15       # higher = more distortion
audio_signal = np.tanh(drive * audio_signal)
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# 3. REVERB — convolve with exponential decay impulse
# print("Applying reverb...")
# reverb_len  = int(sample_rate * 1.5)   # 1.5 second tail
# decay       = np.exp(-4.0 * np.linspace(0, 1, reverb_len))
# impulse     = decay * np.random.randn(reverb_len) * 0.3
# audio_signal = np.convolve(audio_signal, impulse, mode='full')[:len(audio_signal)]
# audio_signal = audio_signal / np.max(np.abs(audio_signal))

# 4. DELAY — rhythmic echo repeat
print("Applying delay...")
bpm         = 135        # EDM tempo
beat_samples = int((60 / bpm) * sample_rate)
delay_signal = np.zeros_like(audio_signal)
delay_signal[beat_samples:] = audio_signal[:-beat_samples] * 0.5   # 1 beat delay at 50%
delay_signal[beat_samples * 2:] += audio_signal[:-beat_samples * 2] * 0.25  # 2nd repeat
audio_signal = audio_signal + delay_signal
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# 5. SIDECHAIN PUMP — volume ducks every beat like a kick drum
print("Applying sidechain pump...")
pump_rate   = bpm / 60   # pumps per second
pump_env    = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * pump_rate * t_full - np.pi / 2))
audio_signal = audio_signal * pump_env
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# 6. HIGH PASS FILTER — cut muddy low end, keep it crisp
print("Applying high pass filter...")
sos         = signal.butter(4, 80, btype='high', fs=sample_rate, output='sos')
audio_signal = signal.sosfilt(sos, audio_signal)
audio_signal = audio_signal / np.max(np.abs(audio_signal))

print("All effects applied")

sf.write("output_musicaha.wav", audio_signal, sample_rate)
print("Done — output_music.wav generated")
print("Export completed successfully.")
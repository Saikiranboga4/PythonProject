import librosa
import numpy as np
import matplotlib.pyplot as plt


y, sr = librosa.load('Sample.mp3', sr=None)

pitch, magnitudes = librosa.piptrack(y=y, sr=sr)

pitch_over_time = []
for t in range(pitch.shape[1]):
    index = magnitudes[:, t].argmax()
    pitch_over_time.append(pitch[index, t]) 

#print(pitch_over_time)

pitch_array = np.array(pitch_over_time)

high_nodes = np.where(pitch_over_time > np.percentile(pitch_over_time, 75))[0]
low_nodes = np.where(pitch_over_time < np.percentile(pitch_over_time, 25))[0]

time_axis = np.linspace(0, len(y) / sr, len(pitch_array))

plt.figure(figsize=(14, 5))
plt.plot(time_axis, pitch_array, label='Pitch', color='steelblue', linewidth=0.8)

plt.scatter(time_axis[high_nodes], pitch_array[high_nodes]
            , color='red', s=10, label='High Nodes', zorder=3)
plt.scatter(time_axis[low_nodes], pitch_array[low_nodes]
            , color='green', s=10, label='Low Nodes', zorder=3)

plt.xlabel("Time (seconds)")
plt.ylabel('Frequency (Hz)')
plt.title('High and Low Nodes Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('pitch_analysis.png')
plt.show()
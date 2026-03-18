"""Quick test: list PyAudio input devices and capture 1s of audio."""
import pyaudio
import numpy as np

pa = pyaudio.PyAudio()
print("=== Input Devices ===")
target_idx = None
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if int(info.get("maxInputChannels", 0)) > 0:
        name = info["name"]
        ch = info["maxInputChannels"]
        rate = int(info["defaultSampleRate"])
        print(f"  [{i}] {name}  ({ch}ch, {rate}Hz)")
        if "usb" in name.lower() or "pnp" in name.lower():
            target_idx = i

if target_idx is None:
    print("ERROR: No USB mic found")
    pa.terminate()
    exit(1)

print(f"\n=== Capturing 1s from device {target_idx} ===")
RATE = 44100
CHUNK = 1024
stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                 input=True, input_device_index=target_idx,
                 frames_per_buffer=CHUNK)
frames = []
for _ in range(int(RATE / CHUNK)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(np.frombuffer(data, dtype=np.float32))

stream.stop_stream()
stream.close()
pa.terminate()

audio = np.concatenate(frames)
print(f"  Samples: {len(audio)}")
print(f"  Duration: {len(audio)/RATE:.2f}s")
print(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")
print(f"  Max: {np.max(np.abs(audio)):.6f}")
print("MIC_TEST_PASSED")

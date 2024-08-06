import soundfile as sf
from faster_whisper import WhisperModel

model_size = "large-v2"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

audio_file = 'test_audio.opus'
data, sample_rate = sf.read(audio_file)
segments, info = model.transcribe(data, beam_size=5)

# segments, info = model.transcribe("test_audio.opus", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

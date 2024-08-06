import io
import soundfile as sf
import whisperx                                                            
whisper_model = whisperx.load_model("large-v2", "cuda", compute_type="float16", language="en", task="transcribe", vad=None)

audio_file = 'test_audio.opus'

data, sample_rate = sf.read(audio_file)

result = whisper_model.transcribe(data, batch_size=16)
transcribed_text = "".join(s["text"] for s in result["segments"])

print(transcribed_text)

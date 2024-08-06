from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import librosa
import numpy as np
import io
import time
import re
import soundfile as sf
import torch

from melo.api import TTS

'''
RTF is good 0.02-0.03, but latency not mouch improved
does not handle acronyms well
'''


app = FastAPI()

t0 = time.time()

model = TTS(language='EN', device='auto')
speaker_ids = model.hps.data.spk2id

elapsed = time.time() - t0
print(f"Loaded in {elapsed:.2f}s")

class TTSRequest(BaseModel):
    text: str
    # Female
    speaker: str = "p273"
    speaker: str = "p335"

    # Male

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return "OK"

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # Text preprocessing
        text = request.text.strip()
        text = re.sub(r'~+', '!', text)
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"(\*[^*]+\*)|(_[^_]+_)", "", text).strip()
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        t0 = time.time()

        buffer = io.BytesIO()
        print(speaker_ids)
        model.tts_to_file(text, speaker_ids['EN-Default'], buffer, speed=1.0, format='wav')
        buffer.seek(0)
        wav_np, sr = sf.read(buffer)

        generation_time = time.time() - t0

        audio_duration = len(wav_np) / sr
        rtf = generation_time / audio_duration
        print(f"Generated in {generation_time:.2f}s")
        print(f"Real-Time Factor (RTF): {rtf:.2f}")

        wav_np = np.clip(wav_np, -1, 1)

        # Resample to 24kHz
        if sr != 24000:
            wav_np_24k = librosa.resample(wav_np, orig_sr=sr, target_sr=24000)
        else:
            wav_np_24k = wav_np

        # Convert to Opus using an in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, wav_np_24k, 24000, format='ogg', subtype='opus')
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/ogg; codecs=opus")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

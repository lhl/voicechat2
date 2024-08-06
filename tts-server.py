from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from TTS.api import TTS
import librosa
import numpy as np
import io
import time
import re
import soundfile as sf
import torch


app = FastAPI()

print('Loading VITS...')
t0 = time.time()
vits_model = 'tts_models/en/vctk/vits'
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_vits = TTS(vits_model).to(device)
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
    return """
    <html>
        <body>
            <style>
            textarea, input { display: block; width: 100%; border: 1px solid #999; margin: 10px 0px }
            textarea { height: 25%; }
            </style>
            <h2>TTS VITS</h2>
            <form method="post" action="/tts">
                <textarea name="text">This is a test.</textarea>
                <input name="speaker" value="p273" />
                <input type="submit" />
            </form>
        </body>
    </html>
    """

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
        wav_np = tts_vits.tts(text, speaker=request.speaker)
        generation_time = time.time() - t0

        audio_duration = len(wav_np) / 22050  # Assuming 22050 Hz sample rate
        rtf = generation_time / audio_duration
        print(f"Generated in {generation_time:.2f}s")
        print(f"Real-Time Factor (RTF): {rtf:.2f}")

        wav_np = np.array(wav_np)
        wav_np = np.clip(wav_np, -1, 1)

        # Resample to 24kHz
        original_sr=22050
        wav_np_24k = librosa.resample(wav_np, orig_sr=original_sr, target_sr=24000)

        # Convert to Opus using an in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, wav_np_24k, 24000, format='ogg', subtype='opus')
        # sf.write(buffer, wav_np, 24000, format='ogg', subtype='opus')
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/ogg; codecs=opus")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

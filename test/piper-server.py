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

'''
https://onnxruntime.ai/docs/install/#python-installs
# CUDA 11
pip install onnxruntime-gpu
# CUDA 12
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

pip install piper

You need to edit the voice.py:
https://github.com/rhasspy/piper/pull/172/commits/e0f71939b9ba325c21262ac38f4a2b120f9b91c6
```
else [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})],
```
if you don't update this then CUDA is like 5X slower than CPU!
See also: https://github.com/rhasspy/piper/issues/343

works but sometimes has weird error:
2024-08-07 01:09:23.661101818 [E:onnxruntime:, sequential_executor.cc:516 ExecuteKernel] Non-zero status code returned while running Reshape node. Name:'Reshape_5227' S
tatus Message: /onnxruntime_src/onnxruntime/core/providers/cpu/tensor/reshape_helper.h:28 onnxruntime::ReshapeHelper::ReshapeHelper(const onnxruntime::TensorShape&, onn
xruntime::TensorShapeVector&, bool) i < input_shape.NumDimensions() was false. The dimension with value zero exceeds the dimension size of the input tensor.

Error: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node. Name:'Reshape_5227' Status Message: /onnxruntime_src/onnxr
untime/core/providers/cpu/tensor/reshape_helper.h:28 onnxruntime::ReshapeHelper::ReshapeHelper(const onnxruntime::TensorShape&, onnxruntime::TensorShapeVector&, bool) i
 < input_shape.NumDimensions() was false. The dimension with value zero exceeds the dimension size of the input tensor.

also 
'''

from piper import PiperVoice


app = FastAPI()

t0 = time.time()

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

# run first to d/l model
# echo 'Welcome to the world of speech synthesis!' | piper   --model en_US-libritts-high -s 0   --output_file welcome.wav

voice = PiperVoice.load('en_US-libritts-high.onnx', 'en_US-libritts-high.onnx.json', use_cuda=use_cuda)
synthesize_args = {
    "speaker_id": 0,
}

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

        audio_generator = voice.synthesize_stream_raw(text, **synthesize_args)
        audio_chunks = list(audio_generator)
        audio_data = b''.join(audio_chunks)
        wav_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        generation_time = time.time() - t0

        audio_duration = len(wav_np) / voice.config.sample_rate
        rtf = generation_time / audio_duration
        print(f"Generated in {generation_time:.2f}s")
        print(f"Real-Time Factor (RTF): {rtf:.2f}")

        wav_np = np.clip(wav_np, -1, 1)

        # Resample to 24kHz
        wav_np_24k = librosa.resample(wav_np, orig_sr=voice.config.sample_rate, target_sr=24000)

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

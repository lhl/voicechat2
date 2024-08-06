import soundfile as sf
import sys
import tempfile
import torch

from abc import ABC, abstractmethod
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Union
from urllib.parse import unquote

app = FastAPI()

class TranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, file, audio_content, **kwargs):
        pass

class TransformersEngine(TranscriptionEngine):
    def __init__(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 400ms
        model_id = "openai/whisper-large-v2"

        # 220ms
        model_id = "distil-whisper/large-v2"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(self, file, audio_content, **kwargs):
        result = self.pipe(audio_content, **kwargs)
        return result["text"], result.get("chunks", [])

class FasterWhisperEngine(TranscriptionEngine):
    def __init__(self):
        from faster_whisper import WhisperModel

        # 350ms
        model_id = "large-v2"
        # 300ms
        model_id = "distil-large-v2"
        # 280ms
        model_id = "distil-medium.en"
        # 300ms
        model_id = "distil-large-v3"
        
        self.model = WhisperModel(model_id, device="cuda", compute_type="float16")

    def transcribe(self, file, audio_content, **kwargs):
        segments, _ = self.model.transcribe(unquote(file.filename), beam_size=5)

        full_text = "".join(segment.text for segment in segments)
        logger.info(full_text)

        return full_text, [{"start": s.start, "end": s.end, "text": s.text} for s in segments]


# For shorter sentences, the regular transformers pipeline seems to be faster than faster-whisper?
'''
try:
    engine = FasterWhisperEngine()
    logger.info("Using FasterWhisperEngine")
except ImportError:
    engine = TransformersEngine()
    logger.info("Using TransformersEngine")
'''
engine = TransformersEngine()
logger.info("Using TransformersEngine")

class TranscriptionResponse(BaseModel):
    text: str


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str = Form(None),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    audio_content = await file.read()
    text, _ = engine.transcribe(audio_content, generate_kwargs={"language": language, "task": "transcribe"})
    response = {"text": text}
    return JSONResponse(content=response, media_type="application/json")


@app.post("/v1/audio/translations", response_model=TranscriptionResponse)
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    # Read the audio file
    audio_content = await file.read()
    text, _ = engine.transcribe(audio_content, generate_kwargs={"task": "translate"})
    response = {"text": text}
    return JSONResponse(content=response, media_type="application/json")


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    temperature: float = Form(0.0),
    temperature_inc: float = Form(0.0),
    response_format: str = Form("json")
):
    # Read the audio file
    audio_content = await file.read()

    temperature += temperature_inc
    
    text, segments = engine.transcribe(
        file,
        audio_content,
        generate_kwargs={
            "temperature": temperature,
            "do_sample": True
        } if isinstance(engine, TransformersEngine) else {
            "beam_size": 5,
            "temperature": temperature
        }
    )
    
    # Prepare the response based on the requested format
    if response_format == "json":
        response = {
            "text": text,
            # "segments": [
            #     {
            #         "start": segment["timestamp"][0],
            #         "end": segment["timestamp"][1],
            #         "text": segment["text"]
            #     }
            #     for segment in result["chunks"]
            # ]
        }
    else:
        response = {"text": text}
    
    return JSONResponse(content=response, media_type="application/json")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama
from TTS.api import TTS
import numpy as np
from scipy.io.wavfile import write, read
import whisperx
import asyncio
import io
import logging
import os
import re
import tempfile
import time
import json
import uuid
import opuslib

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models
print("Loading SRT...")
t0 = time.time()
whisperx_model = whisperx.load_model("large-v2", "cuda", compute_type="float16", language="en")
elapsed = time.time() - t0
print(f"Loaded SRT in {elapsed:.2f}s")

print("Loading LLM...")
t0 = time.time()
llama_model = Llama(model_path="/models/llm/nouse-hermes-llama2/ggml-Hermes-2-step2559-q4_K_M.bin", n_ctx=4096, n_gpu_layers=99)
elapsed = time.time() - t0
print(f"Loaded LLM in {elapsed:.2f}s")

print('Loading vits...')
t0 = time.time()
vits_model = 'tts_models/en/vctk/vits'
tts_vits = TTS(vits_model, gpu=True)
elapsed = time.time() - t0
print(f"Loaded TTS in {elapsed:.2f}s")

class ConversationManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "conversation": [],
            "llm_output_sentences": deque(),
            "current_turn": 0,
            "is_processing": False
        }
        return session_id

    def add_user_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "user", "content": message})
        self.sessions[session_id]["current_turn"] += 1

    def add_ai_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "ai", "content": message})
        self.sessions[session_id]["current_turn"] += 1

conversation_manager = ConversationManager()

async def transcribe_audio(audio_data):
    # Decode Opus to PCM
    opus_decoder = opuslib.Decoder(48000, 1)
    pcm_data = opus_decoder.decode(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        write(temp_file.name, 48000, np.frombuffer(pcm_data, dtype=np.int16))
        temp_file.close()

        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, whisperx.load_audio, temp_file.name)
        result = await loop.run_in_executor(None, whisperx_model.transcribe, audio, 16)

        text = " ".join([s["text"] for s in result["segments"]])
        os.unlink(temp_file.name)
        return text

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = conversation_manager.create_session()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            if conversation_manager.sessions[session_id]["is_processing"]:
                # Handle interruption
                conversation_manager.sessions[session_id]["llm_output_sentences"].clear()
                conversation_manager.sessions[session_id]["is_processing"] = False
                await websocket.send_json({"type": "interrupted"})
            else:
                conversation_manager.sessions[session_id]["is_processing"] = True
                text = await transcribe_audio(data)
                conversation_manager.add_user_message(session_id, text)
                
                # Start LLM and TTS processing
                asyncio.create_task(process_and_stream(websocket, session_id, text))
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")

async def process_and_stream(websocket: WebSocket, session_id, text):
    try:
        # LLM processing
        await generate_llm_response(session_id, text)
        
        # TTS processing and streaming
        await generate_and_stream_tts(websocket, session_id)
    finally:
        conversation_manager.sessions[session_id]["is_processing"] = False

async def generate_llm_response(session_id, text):
    prompt = f"""### Instruction:
{text}

### Input:
This is part of a text-to-speech conversation so please keep answers shorter and in a way that makes sense when spoken aloud in a dialogue.

### Response:
"""
    current_buffer = ""
    for token in llama_model.create_completion(prompt,
                                               max_tokens = 1000,
                                               temperature = 0.8,
                                               top_p = 0.95,
                                               stop=["### Instruction:"],
                                               stream = True):
        current_buffer += token['choices'][0]['text']
        print(token['choices'][0]['text'], end='')

        sentences = re.split(r'(?<=[.!?])\s+', current_buffer)
        if len(sentences) > 1:
            sentence = process_sentence(sentences[0])
            conversation_manager.sessions[session_id]["llm_output_sentences"].append(sentence)
            current_buffer = sentences[1]

        if token['choices'][0]['finish_reason']:
            if current_buffer.strip():
                sentence = process_sentence(current_buffer.strip())
                conversation_manager.sessions[session_id]["llm_output_sentences"].append(sentence)
            conversation_manager.sessions[session_id]["llm_output_sentences"].append("<<STOP>>")

def process_sentence(sentence):
    sentence = re.sub(r'~+', '!', sentence)
    sentence = re.sub(r"\(.*?\)", "", sentence)
    sentence = re.sub(r"(\*[^*]+\*)|(_[^_]+_)", "", sentence)
    sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
    return sentence.strip()

async def generate_and_stream_tts(websocket: WebSocket, session_id):
    speaker = 'p273'
    llm_output_sentences = conversation_manager.sessions[session_id]["llm_output_sentences"]
    
    while True:
        if len(llm_output_sentences) > 0:
            sentence = ""

            while len(sentence) < 25 and llm_output_sentences:
                new_sentence = llm_output_sentences.popleft()
                if new_sentence == '<<STOP>>':
                    if sentence:
                        await tts_generate_and_send(websocket, sentence, speaker)
                    return
                else:
                    sentence += " " + new_sentence

            if sentence:
                await tts_generate_and_send(websocket, sentence, speaker)
        else:
            await asyncio.sleep(0.1)

async def tts_generate_and_send(websocket: WebSocket, sentence, speaker):
    t0 = time.time()
    wav_np = tts_vits.tts(sentence, speaker=speaker)
    elapsed = time.time() - t0
    print(f"Sentence '{sentence}' generated in {elapsed:.2f}s")

    wav_np = np.array(wav_np)
    wav_np_int16 = np.int16(wav_np * 32767)
    
    # Encode to Opus
    opus_encoder = opuslib.Encoder(48000, 1, opuslib.APPLICATION_AUDIO)
    opus_data = opus_encoder.encode(wav_np_int16.tobytes())

    await websocket.send_bytes(opus_data)

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

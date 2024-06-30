import asyncio
import aiohttp
import json
import os
import re
import tempfile
import time
import uuid
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from scipy.io.wavfile import write
import opuslib

# External endpoints
SRT_ENDPOINT = os.getenv("SRT_ENDPOINT", "http://localhost:8001/transcribe")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8002/v1/chat/completions")
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "http://localhost:8003/tts")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        self.sessions[session_id]["conversation"].append({"role": "assistant", "content": message})
        self.sessions[session_id]["current_turn"] += 1

conversation_manager = ConversationManager()

async def transcribe_audio(audio_data):
    # Decode Opus to PCM
    opus_decoder = opuslib.Decoder(48000, 1)
    pcm_data = opus_decoder.decode(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        write(temp_file.name, 48000, np.frombuffer(pcm_data, dtype=np.int16))
        temp_file.close()

        async with aiohttp.ClientSession() as session:
            async with session.post(SRT_ENDPOINT, data={'audio': open(temp_file.name, 'rb')}) as response:
                result = await response.json()

        os.unlink(temp_file.name)
        return result['text']

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
        await generate_llm_response(websocket, session_id, text)
        
        # TTS processing and streaming
        await generate_and_stream_tts(websocket, session_id)
    finally:
        conversation_manager.sessions[session_id]["is_processing"] = False

async def generate_llm_response(websocket: WebSocket, session_id, text):
    conversation = conversation_manager.sessions[session_id]["conversation"]
    
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_ENDPOINT, json={
            "messages": conversation,
            "stream": True
        }) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8').split('data: ')[1])
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['delta'].get('content', '')
                            if content:
                                await process_llm_content(websocket, session_id, content)
                    except json.JSONDecodeError:
                        pass  # Ignore non-JSON lines

async def process_llm_content(websocket: WebSocket, session_id, content):
    sentences = re.split(r'(?<=[.!?])\s+', content)
    for sentence in sentences:
        if sentence:
            processed_sentence = process_sentence(sentence)
            conversation_manager.sessions[session_id]["llm_output_sentences"].append(processed_sentence)
            conversation_manager.add_ai_message(session_id, processed_sentence)

def process_sentence(sentence):
    sentence = re.sub(r'~+', '!', sentence)
    sentence = re.sub(r"\(.*?\)", "", sentence)
    sentence = re.sub(r"(\*[^*]+\*)|(_[^_]+_)", "", sentence)
    sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
    return sentence.strip()

async def generate_and_stream_tts(websocket: WebSocket, session_id):
    llm_output_sentences = conversation_manager.sessions[session_id]["llm_output_sentences"]
    
    while llm_output_sentences:
        sentence = llm_output_sentences.popleft()
        if sentence:
            await tts_generate_and_send(websocket, sentence)

async def tts_generate_and_send(websocket: WebSocket, sentence):
    async with aiohttp.ClientSession() as session:
        async with session.post(TTS_ENDPOINT, json={"text": sentence}) as response:
            audio_data = await response.read()

    # Encode to Opus
    opus_encoder = opuslib.Encoder(48000, 1, opuslib.APPLICATION_AUDIO)
    opus_data = opus_encoder.encode(audio_data)

    await websocket.send_bytes(opus_data)

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

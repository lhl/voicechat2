import asyncio
import aiohttp
import io
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import opuslib
import numpy as np
import traceback

# External endpoints
SRT_ENDPOINT = os.getenv("SRT_ENDPOINT", "http://localhost:8001/inference")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8002/v1/chat/completions")
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "http://localhost:8003/tts")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

SYSTEM = {
    "role": "system",
    "content": "You are a helpful assistant. We are interacting via voice so keep responses concise, no more than to a couple sentences unless the user specifies a longer response."
}

class ConversationManager:
    def __init__(self):
        self.sessions = {}
        self.session_timeout = 3600  # 1 hour timeout for sessions

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "conversation": [SYSTEM],
            "llm_output_sentences": deque(),
            "current_turn": 0,
            "is_processing": False,
            "audio_buffer": b'',  # New: Buffer to accumulate audio data
            "last_activity": time.time()
        }
        return session_id

    def add_user_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "user", "content": message})
        self.sessions[session_id]["current_turn"] += 1
        self.sessions[session_id]["last_activity"] = time.time()

    def add_ai_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "assistant", "content": message})
        self.sessions[session_id]["current_turn"] += 1
        self.sessions[session_id]["last_activity"] = time.time()

    def get_conversation(self, session_id):
        return self.sessions[session_id]["conversation"]

    def clean_old_sessions(self):
        current_time = time.time()
        sessions_to_remove = [
            session_id for session_id, session_data in self.sessions.items()
            if current_time - session_data["last_activity"] > self.session_timeout
        ]
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

    def add_to_audio_buffer(self, session_id, audio_data):
        self.sessions[session_id]["audio_buffer"] += audio_data

    def get_and_clear_audio_buffer(self, session_id):
        audio_data = self.sessions[session_id]["audio_buffer"]
        self.sessions[session_id]["audio_buffer"] = b''
        return audio_data

conversation_manager = ConversationManager()

async def transcribe_audio(audio_data, session_id, turn_id):
    try:
        temp_file_path = f"/tmp/{session_id}-{turn_id}.opus"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio_data)

        # Add a small delay to ensure the file is fully written
        await asyncio.sleep(0.1)

        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', open(temp_file_path, 'rb'), filename=f"/tmp/{session_id}-{turn_id}.opus")
            data.add_field('temperature', "0.0")
            data.add_field('temperature_inc', "0.2")
            data.add_field('response_format', "json")

            async with session.post(SRT_ENDPOINT, data=data) as response:
                result = await response.json()

        # Optionally, you can remove the temporary file here if you don't need it for debugging
        # os.remove(temp_file_path)

        logger.debug(result)
        return result['text']
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = conversation_manager.create_session()
    logger.info(f"New WebSocket connection established. Session ID: {session_id}")
    
    try:
        while True:
            message = await websocket.receive()
            logger.debug(f"Received message: {message}")
            
            if 'bytes' in message:
                audio_data = message['bytes']
                logger.debug(f"Received audio data. Size: {len(audio_data)} bytes")
                conversation_manager.sessions[session_id]["audio_buffer"] = audio_data
            elif 'text' in message:
                logger.debug(f"Received text message: {message['text']}")
                try:
                    data = json.loads(message['text'])
                    logger.debug(f"Parsed JSON data: {data}")
                    if data.get("action") == "stop_recording":
                        logger.info("Stop recording message received. Processing audio...")
                        if conversation_manager.sessions[session_id]["is_processing"]:
                            logger.warning("Interrupting ongoing processing")
                            conversation_manager.sessions[session_id]["llm_output_sentences"].clear()
                            conversation_manager.sessions[session_id]["is_processing"] = False
                            await websocket.send_json({"type": "interrupted"})
                        else:
                            conversation_manager.sessions[session_id]["is_processing"] = True
                            turn_id = conversation_manager.sessions[session_id]["current_turn"]
                            try:
                                audio_data = conversation_manager.sessions[session_id]["audio_buffer"]
                                logger.info(f"Processing audio data. Size: {len(audio_data)} bytes")
                                text = await transcribe_audio(audio_data, session_id, turn_id)
                                if not text:
                                    raise ValueError("Transcription resulted in empty text")
                                logger.info(f"Transcription result: {text}")
                                conversation_manager.add_user_message(session_id, text)
                                
                                await process_and_stream(websocket, session_id, text)
                                
                                await websocket.send_json({"type": "processing_complete"})
                            except Exception as e:
                                logger.error(f"Error during processing: {str(e)}")
                                logger.error(traceback.format_exc())
                                await websocket.send_json({"type": "error", "message": str(e)})
                            finally:
                                conversation_manager.sessions[session_id]["is_processing"] = False
                    else:
                        logger.warning(f"Received unexpected action: {data.get('action')}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from text message: {message['text']}")
            else:
                logger.warning(f"Received message with unexpected format: {message}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        await websocket.close(code=1011, reason=str(e))


async def process_and_stream(websocket: WebSocket, session_id, text):
    try:
        # LLM processing
        await generate_llm_response(websocket, session_id, text)
        
        # TTS processing and streaming
        await generate_and_stream_tts(websocket, session_id)
    finally:
        conversation_manager.sessions[session_id]["is_processing"] = False


async def generate_llm_response(websocket, session_id, text):
    try:
        conversation = conversation_manager.get_conversation(session_id)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_ENDPOINT, json={
                "model": "gpt-3.5-turbo",
                "messages": conversation + [{"role": "user", "content": text}],
                "stream": True
            }) as response:
                accumulated_text = ""
                async for line in response.content:
                    if line:
                        try:
                            line_text = line.decode('utf-8').strip()
                            if line_text.startswith('data: '):
                                data_str = line_text[6:]
                                if data_str.lower() == '[done]':
                                    break
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    content = data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        accumulated_text += content
                                        await websocket.send_json({"type": "text", "content": content})
                                        
                                        # Check if we have a complete sentence or substantial chunk
                                        if content.endswith(('.', '!', '?')) or len(accumulated_text) > 100:
                                            await generate_and_send_tts(websocket, accumulated_text)
                                            accumulated_text = ""
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON: {line_text}")
                        except Exception as e:
                            logger.error(f"Error processing line: {e}")
                
                # Send any remaining text
                if accumulated_text:
                    conversation_manager.sessions[session_id]["llm_output_sentences"].append(accumulated_text)
                    await generate_and_send_tts(websocket, accumulated_text)


                conversation_manager.add_ai_message(session_id, "".join(conversation_manager.sessions[session_id]["llm_output_sentences"]))

    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def generate_and_send_tts(websocket, text):
    async with aiohttp.ClientSession() as session:
        async with session.post(TTS_ENDPOINT, json={"text": text}) as response:
            opus_data = await response.read()
    await websocket.send_bytes(opus_data)


async def process_llm_content(websocket, session_id, content):
    sentences = re.split(r'(?<=[.!?])\s+', content)
    for sentence in sentences:
        if sentence:
            processed_sentence = process_sentence(sentence)
            conversation_manager.sessions[session_id]["llm_output_sentences"].append(processed_sentence)
            conversation_manager.add_ai_message(session_id, processed_sentence)
            logger.debug(f"Processed sentence: {processed_sentence}")

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
            opus_data = await response.read()

    # Send the Opus data directly
    await websocket.send_bytes(opus_data)


@app.get("/")
def read_root():
    return FileResponse("ui/index.html")

# Run session cleanup periodically
'''
@app.on_event("startup")
@app.on_event("shutdown")
async def cleanup_sessions():
    while True:
        conversation_manager.clean_old_sessions()
        await asyncio.sleep(3600)  # Run cleanup every hour
'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

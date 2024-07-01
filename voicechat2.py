import asyncio
import aiohttp
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
                turn_id = conversation_manager.sessions[session_id]["current_turn"]
                try:
                    text = await transcribe_audio(data, session_id, turn_id)
                    if not text:
                        raise ValueError("Transcription resulted in empty text")
                    conversation_manager.add_user_message(session_id, text)
                    
                    # Start LLM and TTS processing
                    await process_and_stream(websocket, session_id, text)
                    
                    # Signal end of processing
                    await websocket.send_json({"type": "processing_complete"})
                except Exception as e:
                    logger.error(f"Error during processing: {str(e)}")
                    logger.error(traceback.format_exc())
                    await websocket.send_json({"type": "error", "message": str(e)})
                finally:
                    conversation_manager.sessions[session_id]["is_processing"] = False
    
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
        conversation = conversation_manager.sessions[session_id]["conversation"]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_ENDPOINT, json={
                "model": "gpt-3.5-turbo",  # This might be ignored by llama.cpp
                "messages": conversation + [{"role": "user", "content": text}],
                "stream": True
            }) as response:
                logger.debug(f"LLM response status: {response.status}")
                async for line in response.content:
                    logger.debug(f"Raw line: {line}")
                    if line:
                        try:
                            line_text = line.decode('utf-8').strip()
                            if line_text.startswith('data: '):
                                data_str = line_text[6:]  # Remove 'data: ' prefix
                                if data_str.lower() == '[done]':
                                    logger.debug("Received [DONE] from LLM")
                                    break
                                data = json.loads(data_str)
                                logger.debug(f"Parsed data: {data}")
                                if 'choices' in data and len(data['choices']) > 0:
                                    content = data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        await process_llm_content(websocket, session_id, content)
                            else:
                                logger.warning(f"Unexpected line format: {line_text}")
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON: {line_text}")
                        except Exception as e:
                            logger.error(f"Error processing line: {e}")
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        logger.error(traceback.format_exc())
        raise


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

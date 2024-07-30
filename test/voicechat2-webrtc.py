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
import traceback
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaRecorder

# External endpoints
SRT_ENDPOINT = os.getenv("SRT_ENDPOINT", "http://localhost:8001/inference")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8002/v1/chat/completions")
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "http://localhost:8003/tts")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Set the logging level for aioice to INFO or WARNING
logging.getLogger('aioice').setLevel(logging.WARNING)

# If you want to silence aiortc logs as well, you can do:
logging.getLogger('aiortc').setLevel(logging.WARNING)

# Keep your application's logger at DEBUG level
logger.setLevel(logging.DEBUG)


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
            "is_processing": False,
            "peer_connection": None,
            "audio_track": None,
            "data_channel": None,
            "audio_buffer": io.BytesIO()  # New: buffer to accumulate audio data
        }
        return session_id

    def set_data_channel(self, session_id, channel):
        self.sessions[session_id]["data_channel"] = channel
        logger.debug(f"Data channel set for session {session_id}")

    def get_data_channel(self, session_id):
        channel = self.sessions[session_id]["data_channel"]
        if channel is None:
            logger.warning(f"Data channel is None for session {session_id}")
        return channel

    def add_user_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "user", "content": message})
        self.sessions[session_id]["current_turn"] += 1

    def add_ai_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "assistant", "content": message})
        self.sessions[session_id]["current_turn"] += 1


conversation_manager = ConversationManager()


async def safe_send(channel, message):
    if channel is None:
        logger.error("Attempted to send message on None channel")
        return False
    try:
        await channel.send(message)
        return True
    except Exception as e:
        logger.error(f"Failed to send message: {str(e)}")
        return False


async def transcribe_audio(audio_data, session_id, turn_id):
    try:
        temp_file_path = f"/tmp/{session_id}-{turn_id}.opus"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio_data)

        await asyncio.sleep(0.1)

        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', open(temp_file_path, 'rb'), filename=f"/tmp/{session_id}-{turn_id}.opus")
            data.add_field('temperature', "0.0")
            data.add_field('temperature_inc', "0.2")
            data.add_field('response_format', "json")

            async with session.post(SRT_ENDPOINT, data=data) as response:
                result = await response.json()

        os.remove(temp_file_path)

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
    ice_candidates_queue = []
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data['type'] == 'offer':
                offer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
                pc = RTCPeerConnection()
                conversation_manager.sessions[session_id]["peer_connection"] = pc
                
                @pc.on("datachannel")
                def on_datachannel(channel):
                    conversation_manager.sessions[session_id]["data_channel"] = channel
                    @channel.on("message")
                    async def on_message(message):
                        if conversation_manager.sessions[session_id]["is_processing"]:
                            conversation_manager.sessions[session_id]["llm_output_sentences"].clear()
                            conversation_manager.sessions[session_id]["is_processing"] = False
                            await channel.send(json.dumps({"type": "interrupted"}))
                        else:
                            conversation_manager.sessions[session_id]["is_processing"] = True
                            await process_audio_message(message, session_id, channel)
                
                @pc.on("track")
                def on_track(track):
                    if track.kind == "audio":
                        conversation_manager.sessions[session_id]["audio_track"] = track
                
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                await websocket.send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp,
                })

                # Process any queued ICE candidates
                for candidate_data in ice_candidates_queue:
                    candidate = RTCIceCandidate(
                        sdpMid=candidate_data['sdpMid'],
                        sdpMLineIndex=candidate_data['sdpMLineIndex'],
                        candidate=candidate_data['candidate']
                    )
                    await pc.addIceCandidate(candidate)
                ice_candidates_queue.clear()
            
            elif data['type'] == 'ice-candidate':
                if conversation_manager.sessions[session_id]["peer_connection"] is not None:
                    candidate_parts = data['candidate']['candidate'].split()
                    candidate = RTCIceCandidate(
                        component=int(candidate_parts[1]),
                        foundation=candidate_parts[0],
                        protocol=candidate_parts[2],
                        priority=int(candidate_parts[3]),
                        ip=candidate_parts[4],
                        port=int(candidate_parts[5]),
                        type=candidate_parts[7],
                        sdpMid=data['candidate'].get('sdpMid'),
                        sdpMLineIndex=data['candidate'].get('sdpMLineIndex')
                    )
                    await conversation_manager.sessions[session_id]["peer_connection"].addIceCandidate(candidate)
                else:
                    # Queue the ICE candidate if peer connection is not yet initialized
                    ice_candidates_queue.append(data['candidate'])
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket endpoint: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        if session_id in conversation_manager.sessions:
            if conversation_manager.sessions[session_id]["peer_connection"]:
                await conversation_manager.sessions[session_id]["peer_connection"].close()
            del conversation_manager.sessions[session_id]
        await websocket.close()


async def process_audio_message(message, session_id, channel):
    try:
        logger.debug(f"Processing audio message for session {session_id}")
        channel = conversation_manager.get_data_channel(session_id)
        
        if channel is None:
            raise ValueError(f"DataChannel is None for session {session_id}")

        if isinstance(message, str) and message == "RECORDING_COMPLETE":
            # Process the accumulated audio data
            audio_data = conversation_manager.sessions[session_id]["audio_buffer"].getvalue()
            conversation_manager.sessions[session_id]["audio_buffer"] = io.BytesIO()  # Reset buffer

            turn_id = conversation_manager.sessions[session_id]["current_turn"]
            text = await transcribe_audio(audio_data, session_id, turn_id)
            
            conversation_manager.add_user_message(session_id, text)
            
            await generate_llm_response(channel, session_id, text)
            await generate_and_stream_tts(channel, session_id)
            
            await safe_send(channel, json.dumps({"type": "processing_complete"}))
        else:
            # Accumulate audio data
            conversation_manager.sessions[session_id]["audio_buffer"].write(message)
            await safe_send(channel, json.dumps({"type": "audio_chunk_received"}))

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error(traceback.format_exc())
        if channel:
            await safe_send(channel, json.dumps({"type": "error", "message": str(e)}))
    finally:
        if isinstance(message, str) and message == "RECORDING_COMPLETE":
            conversation_manager.sessions[session_id]["is_processing"] = False


async def generate_llm_response(channel, session_id, text):
    try:
        conversation = conversation_manager.sessions[session_id]["conversation"]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_ENDPOINT, json={
                "model": "gpt-3.5-turbo",
                "messages": conversation + [{"role": "user", "content": text}],
                "stream": True
            }) as response:
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
                                        await process_llm_content(channel, session_id, content)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON: {line_text}")
                        except Exception as e:
                            logger.error(f"Error processing line: {e}")
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def process_llm_content(channel, session_id, content):
    sentences = re.split(r'(?<=[.!?])\s+', content)
    for sentence in sentences:
        if sentence:
            processed_sentence = process_sentence(sentence)
            conversation_manager.sessions[session_id]["llm_output_sentences"].append(processed_sentence)
            conversation_manager.add_ai_message(session_id, processed_sentence)
            await channel.send(json.dumps({"type": "llm_output", "content": processed_sentence}))

def process_sentence(sentence):
    sentence = re.sub(r'~+', '!', sentence)
    sentence = re.sub(r"\(.*?\)", "", sentence)
    sentence = re.sub(r"(\*[^*]+\*)|(_[^_]+_)", "", sentence)
    sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
    return sentence.strip()

async def generate_and_stream_tts(channel, session_id):
    llm_output_sentences = conversation_manager.sessions[session_id]["llm_output_sentences"]
    
    while llm_output_sentences:
        sentence = llm_output_sentences.popleft()
        if sentence:
            audio_data = await tts_generate(sentence)
            pc = conversation_manager.sessions[session_id]["peer_connection"]
            audio_track = conversation_manager.sessions[session_id]["audio_track"]
            if pc and audio_track:
                audio_track.put_packet(audio_data)
            await channel.send(json.dumps({"type": "tts_complete", "sentence": sentence}))

async def tts_generate(sentence):
    async with aiohttp.ClientSession() as session:
        async with session.post(TTS_ENDPOINT, json={"text": sentence}) as response:
            return await response.read()

@app.get("/")
def read_root():
    return FileResponse("ui/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

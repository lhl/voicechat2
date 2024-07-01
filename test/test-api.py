import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Read your audio file
        with open("test_audio.opus", "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Send audio data
        await websocket.send(audio_data)
        
        # Receive and process responses
        output_data = b''
        while True:
            response = await websocket.recv()
            if isinstance(response, bytes):
                output_data += response
            elif isinstance(response, str):
                response_json = json.loads(response)
                if response_json.get("type") == "processing_complete":
                    break
        
        # Save response to file
        with open("response_audio.opus", "wb") as response_file:
            response_file.write(output_data)
        
        print("Response received and saved as response_audio.opus")

asyncio.get_event_loop().run_until_complete(test_websocket())

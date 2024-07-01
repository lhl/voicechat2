import asyncio
import websockets

async def test_websocket():
    uri = "ws://localhost:8000/ws"  # Adjust if your API server is on a different host/port
    async with websockets.connect(uri) as websocket:
        # Read your audio file
        with open("test_output.opus", "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Send audio data
        await websocket.send(audio_data)
        
        # Receive response
        response = await websocket.recv()
        
        # Save response to file
        with open("response_audio.opus", "wb") as response_file:
            response_file.write(response)
        
        print("Response received and saved as response_audio.opus")

asyncio.get_event_loop().run_until_complete(test_websocket())

from collections import deque
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama
from TTS.api import TTS
import numpy as np
from scipy.io.wavfile import write

import asyncio
import io
import logging
import os
import os.path
import re
import tempfile
import time
import whisperx


# Load Server
app = FastAPI()
app.mount("/ui", StaticFiles(directory="static"), name="static")

# Load SRT (we force English for slightly less latency)
t0 = time.time()
whisperx_model = whisperx.load_model("large-v2", "cuda", compute_type="float16", language="en")
elapsed = time.time() - t0
print(f"Loaded SRT in {elapsed:.2f}s")

# Load LLM
t0 = time.time()
llama_model = Llama(model_path="/models/llm/nouse-hermes-llama2/ggml-Hermes-2-step2559-q4_K_M.bin", n_ctx=4096, n_gpu_layers=99)
elapsed = time.time() - t0
print(f"Loaded LLM in {elapsed:.2f}s")

# Load TTS
print('Loading vits...')
t0 = time.time()
vits_model = 'tts_models/en/vctk/vits'
tts_vits = TTS(vits_model, gpu=True)
elapsed = time.time() - t0
print(f"Loaded TTS in {elapsed:.2f}s")

'''
We can try to streamline further:
[ ] LLM and TTS interleaving per sentence, but might be slower due to context switching!
'''
llm_output_sentences = deque()

# Routes
@app.get("/")
def read_root():
    return FileResponse("ui/index.html")

@app.post("/talk")
async def talk(audio_data: UploadFile = File(...)):
    ### Transcribe 
    t0 = time.time()
    global whisperx_model

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    data = await audio_data.read()
    temp_file.write(data)
    temp_file.close()
    print(temp_file.name)

    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, whisperx.load_audio, temp_file.name)
    result = await loop.run_in_executor(None, whisperx_model.transcribe, audio, 16)

    out = {}
    out["language"] = result["language"]
    text = ""
    for s in result["segments"]:
        text += s["text"]
    out["text"] = text
    print(out)
    os.unlink(temp_file.name)

    elapsed = time.time() - t0
    print(f"Transcribed in {elapsed:.2f}s: {text}")


    ### LLM
    t1 = time.time()
    global llama_model
    global llm_output_sentences

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

        # Check if we have a sentence
        sentences = re.split(r'(?<=[.!?])\s+', current_buffer)  # Splitting the text into sentences
        if len(sentences) > 1:
            # Process
            sentence = sentences[0]
            # Replace Tildes
            sentence = re.sub(r'~+', '!', sentence)
            # Get rid of stuff in parentheses
            sentence = re.sub(r"\(.*?\)", "", sentence)
            # We are going to get rid of emotes * and _
            sentence = re.sub(r"(\*[^*]+\*)|(_[^_]+_)", "", sentence)
            # Remove unicode symbols
            sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
            sentence = sentence.strip()
            llm_output_sentences.append(sentence)

            current_buffer = sentences[1]

        if token['choices'][0]['finish_reason']:
            if(current_buffer.strip()):
                llm_output_sentences.append(current_buffer.strip())
            llm_output_sentences.append("<<STOP>>")

    elapsed = time.time() - t1
    print(f"LLM response in {elapsed:.2f}s")

    ### TTS
    t2 = time.time()
    global tts_vits
    speaker = 'p273'
    async def audio_generator(): 
        done = False
        loop = 0
        while not done:
            print(llm_output_sentences)
            if len(llm_output_sentences) > 0:
                sentence = ""

                # We need to check that we have a long enough sentence
                while len(sentence) < 25 and not done:
                    new_sentence = llm_output_sentences.popleft()
                    if new_sentence == '<<STOP>>':
                        print("STOP found.")
                        done = True
                    else: 
                        sentence += new_sentence

                if done and not sentence:
                    break

                t2 = time.time()
                wav_np = tts_vits.tts(sentence, speaker=speaker)
                elapsed = time.time() - t2
                print(f"Sentence '{sentence}' generated in {elapsed:.2f}s")

                wav_np = np.array(wav_np)
                wav_np_int16 = np.int16(wav_np * 32767)
                wav_bytes = io.BytesIO()
                write(wav_bytes, 22050, wav_np_int16)
                wav_bytes.seek(0)

                yield wav_bytes.read()  # Stream the audio
            else:
                loop += 1
                if loop > 1000:
                    break
                asyncio.sleep(0.001)

    return StreamingResponse(audio_generator(), media_type="audio/wav")

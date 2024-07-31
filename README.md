# voicechat2
A local SRT/LLM/TTS Voicechat using Websockets

[![Watch the video](http://img.youtube.com/vi/j_yMp0uCo_Y/0.jpg)](http://www.youtube.com/watch?v=j_yMp0uCo_Y)

# Install
These instructions are for Ubuntu LTS and assume you've [setup your ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html) or [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) already.

I recommend you use [conda](https://docs.conda.io/en/latest/) or (my preferred), [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for environment management. It will make your life easier.

## System Prereqs
```
sudo apt update

# Not strictly required but helpers we use
sudo apt install byobu curl wget

# Audio processing
sudo apt install espeak-ng ffmpeg libopus0 libopus-dev 
```

## Checkout code 
```
# Create env
mamba create -y -n voicechat2 python=3.11

# Setup
mamba activate voicechat2
git clone https://github.com/lhl/voicechat2
cd voicechat2
pip install -r requirements.txt
```

## whisper.cpp
```
# Build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
# AMD version
# -DGGML_HIP_UMA=ON to work with APUs (but hurts dGPU perf)
GGML_HIPBLAS=1 make -j 
# Nvidia version
GGML_CUDA=1 make -j 

# Get model - large-v2 is 3094 MB
bash ./models/download-ggml-model.sh large-v2
# Quantized version - large-v2-q5_0 is  1080MB
# bash ./models/download-ggml-model.sh large-v2-q5_0

# If you're going to go to the next instruction
cd ..
```

## llama.cpp
```
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
# AMD version
make GGML_HIPBLAS=1 -j 
# Nvidia version
make GGML_CUDA=1 -j 

# Grab your preferred GGUF model
wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

# If you're going to go to the next instruction
cd ..
```

## TTS
```
mamba activate voicechat2
pip install TTS
```

## StyleTTS2
```
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
pip install -r requirements.txt
pip install phonemizer

# Download the LJSpeech Model
# https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main
# https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main
pip install huggingface_hub
huggingface-cli download --local-dir . yl4579/StyleTTS2-LJSpeech
```

We include some extra convenience scripts for launching:
```
run-voicechat2.sh - on your GPU machine, tries to launch all servers in separate byobu sessions
remote-tunnel.sh - connect your GPU machine to a jump machine
local-tunnel.sh - connect to the GPU machine via a jump machine
```


# Other AI Voicechat Projects

## webrtc-ai-voice-chat
The demo shows a fair amount of latency (~10s) but this project isn't so different (uses WebRTC not websockets) from voicechat2 (HF Transformers, Ollama)
- https://github.com/lalanikarim/webrtc-ai-voice-chat
- Apache 2.0

## june
A console-based local client (HF Transformers, Ollama, Coqui TTS, PortAudio)
- https://github.com/mezbaul-h/june
- MIT

## GlaDOS
This is a very responsive console-based local-client app that also has VAD and interruption support, plus a really clever hook! (whisper.cpp, llama.cpp, piper, espeak)
- https://github.com/dnhkng/GlaDOS
- MIT

## local-talking-llm
Another console-based local client, more of a proof of concept but with w/ blog writeup.
- https://github.com/vndee/local-talking-llm
- https://blog.duy.dev/build-your-own-voice-assistant-and-run-it-locally/
- MIT

## BUD-E - natural_voice_assistant
Another console-based local client (FastConformer, HF Transformers, StyleTTS2, espeak)
- https://github.com/LAION-AI/natural_voice_assistant
- MIT

## LocalAIVoiceChat
KoljaB has a number of interesting projects around console-based local clients like [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT), [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS), [Linguflex](https://github.com/KoljaB/Linguflex), etc. (faster_whisper, llama.cpp, Coqui XTTS)
- https://github.com/KoljaB/LocalAIVoiceChat
- NC (Coqui Model License)

## rtvi-web-demo
This is *not* a local voicechat client, but it does have a neat WebRTC front-end, so might be worth poking around into (Vite/React, Tailwind, Radix)
- https://github.com/rtvi-ai/rtvi-web-demo

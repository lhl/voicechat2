# voicechat2
A local SRT/LLM/TTS Voicechat using Websockets

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
GGML_HIPBLAS=1 make -j 
# Nvidia version
GGML_CUDA=1 make -j 

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


# Other Audio Projects

- https://github.com/dnhkng/GlaDOS
- https://github.com/LAION-AI/natural_voice_assistant
- https://github.com/KoljaB/RealtimeSTT
- https://github.com/KoljaB/RealtimeTTS

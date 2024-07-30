#!/bin/bash

# Check if byobu is installed
if ! command -v byobu &> /dev/null; then
    echo "byobu is not installed. Please install it and try again."
    exit 1
fi

# Function to create a new window and run a command
create_window() {
    local window_name=$1
    local command=$2
    byobu new-window -n "$window_name"
    byobu send-keys -t "$window_name" "$command" C-m
}

# Create a new byobu session named 'voicechat2' or attach to it if it already exists
byobu new-session -d -s voicechat2

# FastAPI server (with Mamba activation)
create_window "voicechat2" "mamba activate voicechat2 && uvicorn voicechat2:app --host 0.0.0.0 --port 8000 --reload"

# SRT server (whisper.cpp)
create_window "srt" "whisper.cpp/server -m whisper.cpp/models/ggml-large-v2.bin -pr --convert --host 127.0.0.1 --port 8001"
# You could shave off 20% inference time w/ the Q5 quant
# create_window "srt" "whisper.cpp/server -m whisper.cpp/models/ggml-large-v2-q5_0.bin -pr --convert --host 127.0.0.1 --port 8001"

# LLM server (llama.cpp)
create_window "llm" "llama.cpp/llama-server --host 127.0.0.1 --port 8002  -m llama.cpp/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -ngl 99"

# TTS server (with Mamba activation)
create_window "tts" "mamba activate voicechat2 && python tts-server.py"

# Attach to the session
byobu attach-session -t voicechat2

echo "Voice chat system is now running in byobu session 'voicechat2'."
echo "To attach to the session, use: byobu attach -t voicechat2"
echo "To detach from the session, use: Ctrl-a d"

# voicechat2
Local SRT/LLM/TTS Voicechat


# Install
These instructions are for Ubuntu LTS and assume you've [setup your ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html) or [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) already.

I recommend you use [conda](https://docs.conda.io/en/latest/) or (my preferred), [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for environment management. It will make your life easier.

System prereqs
```
sudo apt update

# Not scrictly required but our run scripts use this for managing all the servers
sudo apt install byobu

# Audio processing
sudo apt install espeak-ng ffmpeg libopus0 libopus-dev
```

Then our python
```
# Create env
mamba create -y -n voicechat2 python=3.11

# Setup
git clone https://github.com/lhl/voicechat2

```


```
byobu
ffmpeg
espeak-ng
```

git@github.com:lhl/voicechat2.git

- WebSocket/WebRTC


```
sudo apt install
sudo apt install update
sudo apt install libopus0 libopus-dev
```

```
GGML_HIPBLAS=1 make -j
```



# Other Projects

- https://github.com/dnhkng/GlaDOS
- https://github.com/LAION-AI/natural_voice_assistant
- https://github.com/KoljaB/RealtimeSTT
- https://github.com/KoljaB/RealtimeTTS

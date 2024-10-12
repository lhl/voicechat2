https://www.hackster.io/lhl/voicechat2-local-ai-voice-chat-4c48f2

Published July 31, 2024 © Apache-2.0

# voicechat2: Local AI Voice Chat
A fast AI voicechat system that runs entirely locally (SRT/LLM/TTS) via WebSockets. Compatible with AMD GPUs.

## Team
lhl

## Things used in this project

### Hardware components
AMD Radeon Pro W7900 GPU	

### Software apps and online services
AMD ROCm™ Software	
whisper.cpp
llama.cpp
Coqui TTS
		

## Story

This project is an end-to-end AI voice chat system developed for the [AMD Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023/). While it's not quite as far along as I planned from my [original submission goals](https://www.hackster.io/contests/amd2023/hardware_applications/16885), the project, which I've dubbed [voicechat2](https://github.com/lhl/voicechat2), is still (IMO at least) pretty interesting, being the only WebSocket implementation of a fully local (voice-to-voice) AI chat system that I'm aware of. It's also pretty performant. Here's a demo of it in action:

https://www.youtube.com/watch?v=j_yMp0uCo_Y

**A demo of voicechat2 in action.**


While I've been seeing more voice-to-voice LLM chat demos lately, there are far fewer that run fully locally, and (AFAIK) none with WebSocket support (for easily running a local server on remote devices or for Internet connect serving). Here's a few of the highlights:

- Apache 2.0 licensed - anyone can build and extend on this!
- WebSocket-based - I built a basic Web UI for testing, but the server itself (while a bit under-documented, but see the code itself for the various response codes) is theoretically compatible with any WebSocket client. This is simpler than WebRTC implementations (no STUN/TURN servers required for example, although I did try implementing a WebRTC server as well - see the test folder for various half-baked/alternative implementations). Almost all other other local AI voicechat projects I could find are console-based apps, which while cool, don't allow being run remotely like voicechat2 does.
- Opus audio - I use Opus instead of WAV files for additional bandwidth efficiency (which can also lower latency when the network is a bottleneck).
- Good response time - while there are some faster inference options that aren't compatible across multiple hardware platforms, the out of the box backends I've chosen are still relatively performant, I do some optimizations like sentence splitting for LLM and TTS, and audio and text generation interleaving during output that lowers our voice-to-voice latency without affecting prosody or voice output quality significantly.
- AMD GPU compatibility - As this is a submission for an AMD contest, I've spent a fair amount of time exploring software compatibility and making sure this works on AMD cards.

### What works on AMD?
Before starting with the project proper I wanted to see what I could get working on AMD GPUs. I've previously done some writeups on AMD compatibility with some consumer cards I had access to:

- Notes over the course of 2023, early 2024 tracking AI/ML compatibility: https://llm-tracker.info/howto/AMD-GPUs
- In January 2024 I did a comparison of top-end RDNA3 and 3090/4090 cards for LLM inference: https://www.reddit.com/r/LocalLLaMA/comments/191srof/amd_radeon_7900_xtxtx_inference_performance/
- In mid-February, I did a similar comparison with LLM training: https://www.reddit.com/r/LocalLLaMA/comments/1atvxu2/current_state_of_training_on_amd_radeon_7900_xtx/

For the W7900, which uses the same RNDA3 / Navi 31 / gfx1100 architecture/chip as the 7900 XT/XTX's I ended up doing a fair amount of work reviewing the state of training, inference, and other AI library compatibility. This is up-to-date to around June 2024:
- https://llm-tracker.info/W7900-Pervasive-Computing-Project

I also did a fair bit of additional benchmarking for this project, testing not just LLM performance but also SRT (Whisper implementations) and TTS options:
- https://github.com/AUGMXNT/speed-benchmarking/
- https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1652827441

Probably the most interesting thing here was looking at how the performance compared for Whisper (STT) and inferencing with llama.cpp (which is the highest performance bs=1 LLM option for RDNA3). Sadly, faster-whisper is not available for AMD GPUs as it is up to 5X faster vs HF Transformers performance.

![Some performance numbers](https://hackster.imgix.net/uploads/attachments/1729614/screenshot_2024-06-30_at_8_37_13pm_0arL7TM3WA.png?auto=compress%2Cformat&w=1280&h=960&fit=max)

Most recently, I also did a writeup on WandB comparing Axolotl, Torchtune, and Unsloth performance with the W7900 and 7900 XTX in the mix: https://wandb.ai/augmxnt/train-bench/reports/torchtune-vs-axolotl-vs-unsloth-Trainer-Comparison--Vmlldzo4MzU3NTAx

- *[Unsloth](https://github.com/unslothai/unsloth) is the fastest option, but currently incompatible with AMD since it requires xformers (and maybe some Triton fixes).*

One of the big advantages of running local models is that besides choosing from a huge number of fine-tuned models, you can also make your own for your specific use-cases.

### Implementing the Voice Assistant
For my previous SRT experiments/implementations, I've used [whisperX](https://github.com/m-bain/whisperX), which is a fast and fully featured SRT/STT implementation, however it depends on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) which depends on [CTranslate2](https://github.com/OpenNMT/CTranslate2) and there is sadly, [no ROCm compatible implementation](https://github.com/OpenNMT/CTranslate2/issues/1072).

For now, here are the inferencing backends I've chosen:
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - this is ROCm accelerated and also supports quantizations (eg, the Q5 version uses 1/3 the memory of the standard GGML model; it uses only about 1GB of memory for a large-v2 Q5), and has an OpenAI API compatible server endpoint.
- [llama.cpp](https://github.com/ggerganov/llama.cpp/) - from my previous testing, while still under-optimized, this was still by far the the fastest bs=1 inference option available for ROCm, and also comes with an OpenAI API compatible server endpoint.
- [Coqui TTS](https://github.com/coqui-ai/TTS) - I also implemented a [StyleTTS2](https://github.com/yl4579/StyleTTS2) (the highest ranking open TTS on [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)) version, which is actually also has a better RTF on AMD GPUs than TTS VITS, but I like the various voice options included w/ the Coquit TTS's VITS package better vs StyleTTS2's default LJSpeech model.

These open source components (and the associated models) form the backbone of my local voice assistant. For the demo at the top of the page, I used:

- [whisper ggml-large-v2](https://huggingface.co/ggerganov/whisper.cpp/tree/main) (less hallucinations than v3)
- [Llama 3 8B Instruct (Q4_K_M)](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/tree/main) - a small but all-around very capable conversational model
- tts_models/en/vctk/vits - pre-trained [Coqui TTS](https://github.com/coqui-ai/TTS) models with a wide variety of styles to choose from (there's no good list of speakers, so you'll just have to try them for yourself)

When running with the Q5 Whisper quant, the Q4_K_M LLama 3 8B, and TTS VCTK VITS, you can fit all the models in about 10GB of VRAM. Of course, with 48GB of VRAM, the W7900 can run much larger/more capable LLMs (even a [70B quant](https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/tree/main)), but at the cost of higher latency. If you're looking for something a bit better than Llama 3/3.1 8B Instruct, [Mistral Nemo](https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF) (12B) might be your best bet at the moment.

(One thing that is also worth noting is that while on a W7900 the voice-to-voice latency is usually about 1-1.5 seconds, which is totally acceptable, but the same setup can run with as low as 500ms on an RTX 4090. If you swap in [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) (which requires CUDA for CTranslate2) and a [Distil-Whisper](https://github.com/huggingface/distil-whisper) version of large-v2, you can get as low as 300ms of voice-to-voice latency on the 4090). This is approaching SOTA latency performance (and similar to [GPT4o's voice latency claims](https://openai.com/index/hello-gpt-4o/)) without requiring a native-multimodal/voice-encoding LLM!

![voicechat2 web UI](https://hackster.imgix.net/uploads/attachments/1741949/screenshot_from_2024-07-31_16-30-42_ShDWpv11CM.png?auto=compress%2Cformat&w=1280&h=960&fit=max)

### Usage
As mentioned, the code for voicechat2 is open sourced under an Apache 2.0 license and is available here: https://github.com/lhl/voicechat2

While the code quality is a bit rough at release (could definitely do with some cleanup/refactoring) it does the basics and a few neat things to boot (like having a variety of latency and performance metrics built in). I've implemented a number of queues (sentence-based splitting/queuing on the server and client side) as well to keep interactivity high.

While it's not quite "plug and play", it also shouldn't be too hard for anyone with some Linux/AI software experience to set up. The README has a full step-by-step for setting up inference on an Ubuntu LTS system, and there is a run run-voicechat2.sh helper script that will spin up all the separate services in a [byobu](https://www.byobu.org/) session once everything is set up. Still, at this point, if you are not handy with the shell and Python, you may run into issues.

Also, while you could technically run this (slowly) on CPU, GPUs are recommended. And while it's probably possible to run this on OS X or Windows (WSL would probably be the preferred route I'd guess), I've only been testing/targeting Linux.

Rather than having a second, potentially outdated doc, I'll just link directly to the installation instructions in the README:
- https://github.com/lhl/voicechat2/blob/main/README.md#install

For those with a working [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)/[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) system and [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or my recommendation, [Mamba](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install)), you should be able to get this up running in 10-15 minutes depending on how fast your PC is (for compiling whisper.cpp and llama.cpp) and your network (for downloading the dependencies and models).

If your hardware is weaker, you could switch to running smaller models (eg Whisper Medium or Small, and a sub-7B model like [Phi 3.1 mini](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct), [Zamba2 2.7B](https://huggingface.co/Zyphra/Zamba2-2.7B), or if you have the memory, [DeepSeek V2 Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)).

## Future Work
- Faster Whisper and WhisperX has voice audio detection (VAD) support, as well as running much faster than whisper.cpp. I'd like to implement multiple SRT engine support, but irrespective I think that good VAD is something that would be high on the implementation list. It might be worth running VAD on the client or as a buffer via the websocket to be able to have continuous speech.
- Much of the groundwork (timers, etc) has been laid for gracefully handling interruptions. With good VAD and a bit of math, it should be possible to handle interruptions and pass back when exactly the interruption occurred and updating the session appropriately.
- It would be interesting to look at ways to more easily package all of these models into a one-click experience. Maybe by polishing things up and making it work with [Pinokio](https://pinokio.computer/) or delivery entirely via WebASM/WebGPU would be options.
- I'm still interested in playing around with integrating open source avatars (eg w/ Godot), and extending the basic voice functionality with RAG, function calling, etc, but for voicechat2, I think it'd be better to focus on cleaning up the code base, lowering latency, and making it able to cleanly interface with additional functionality in a modular fashion, rather than hacking things on willy-nilly.
- A lot of my [time and energy is focused on training multilingual (JA/EN) models](https://huggingface.co/shisa-ai) at the moment, so have a voice model that can code-switch between those is also high on my R&D wish list.

OK, well, this write-up is getting a bit long-winded, but hopefully it didn't drag too much and covered all the bases on design, implementation, and the future of the project. Feel free to drop any feedback or questions in the comments here, or issues/PRs to the Github repo directly.

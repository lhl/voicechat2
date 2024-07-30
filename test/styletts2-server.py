import torch
import random
import numpy as np
import yaml
from munch import Munch
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import phonemizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import io
from scipy.io import wavfile
import opuslib
import soundfile as sf

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)

# Add StyleTTS2 path
import sys
import os
styletts2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'StyleTTS2'))
sys.path.insert(0, styletts2_path)

# Import necessary modules
from models import *
from utils import *
from text_utils import TextCleaner

# Initialize device and other components
device = 'cuda' if torch.cuda.is_available() else 'cpu'
textcleaner = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

# Initialize mel spectrogram transform
to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

# Load configuration
config = yaml.safe_load(open("StyleTTS2/Models/LJSpeech/config.yml"))

# Load models
ASR_config = config.get('ASR_config', False)
ASR_config = 'StyleTTS2/' + ASR_config
ASR_path = config.get('ASR_path', False)
ASR_path = 'StyleTTS2/' + ASR_path
text_aligner = load_ASR_models(ASR_path, ASR_config)


F0_path = config.get('F0_path', False)
F0_path = 'StyleTTS2/' + F0_path
pitch_extractor = load_F0_models(F0_path)

from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
BERT_path = 'StyleTTS2/' + BERT_path
plbert = load_plbert(BERT_path)

model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# Load model parameters
params_whole = torch.load("StyleTTS2/Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model[key].load_state_dict(new_state_dict, strict=False)

_ = [model[key].eval() for key in model]

# Initialize sampler
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
    clamp=False
)

# Helper functions
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

# Inference function
def inference(text, noise, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    text = text.replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise,
              embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
              embedding_scale=embedding_scale).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()

# FastAPI app
app = FastAPI()

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        noise = torch.randn(1, 1, 256).to(device)
        wav = inference(request.text, noise, diffusion_steps=5, embedding_scale=1)

        # Ensure the audio is in the correct range (-1 to 1)
        wav = np.clip(wav, -1, 1)

        # Print debug information
        print(f"Audio shape: {wav.shape}")
        print(f"Audio min: {wav.min()}, max: {wav.max()}")
        print(f"Intended sample rate: 24000")
                                        
                        
        # Convert to bytes using an in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, wav, 24000, format='ogg', subtype='opus')
        buffer.seek(0)

        # Read back the file to check its properties
        buffer.seek(0)
        with sf.SoundFile(buffer) as sf_file:
            print(f"Actual sample rate: {sf_file.samplerate}")
            print(f"Channels: {sf_file.channels}")
            print(f"Format: {sf_file.format}")
            print(f"Subtype: {sf_file.subtype}")

        return Response(content=buffer.getvalue(), media_type="audio/ogg; codecs=opus")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)

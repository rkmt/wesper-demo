import sys
sys.path.append("libs/FastSpeech2")

import numpy as np
import librosa
import soundfile as sf
import json
import torch
import yaml
import argparse
import os

import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# FastSpeech2
from model import FastSpeech2
from utils.model import get_vocoder
import utils.tools
print("##### utils.tools.device", utils.tools.device)
import model.modules
print("#### model.moculde.device", model.modules.device)
from utils.tools import pad_2D
from utils.model import vocoder_infer
# HuBERT
from libs.hubert.model import Hubert, URLS, HubertSoft
# Hifigan
import hifigan


def load_fastspeech2(configs, checkpoint_path=None, device='cuda'):
    (preprocess_config, model_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if checkpoint_path:
        print("### loading FastSpeech2", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)
        model.load_state_dict(ckpt["model"], strict=True)

    model = model.to(device)
    model.eval()
    model.requires_grad_ = False
    return model

def units2wav(units, fs2model, vocoder, model_config, preprocess_config, device='cuda', d_targets=None, p_targets=None, e_targets=None):
    ''' units: [N, D=256] 
        fs2model: Unit-FastSpeech2
        vocoder: HiFi-GAN
    '''
    print("### units2wav", "pitch", p_targets) 
    if type(units) == torch.Tensor:
        if len(units.shape) == 3:
            units = units.squeeze(0)
        units = units.detach().cpu().numpy()
        assert len(units.shape) == 2
    units = pad_2D([units])
    #units = torch.from_numpy(units).long().to(device)
    units = torch.from_numpy(units).to(device)
    speakers = torch.tensor([0], device=device) #batch_t[2]
    max_src_len = units.shape[1]
    src_lens = torch.tensor([max_src_len], device=device) #batch_t[4]

    with torch.no_grad(): # predict MELs from UNITs by FastSpeech2
        #print("## device", speakers.device, units.device, src_lens.device)
        # added rkmt 2023.7.20
        if d_targets is None:
            d_targets = torch.tensor([1]*(units.shape[1]), device=device).unsqueeze(0) # use fixed duration
        if p_targets is not None:
            p_targets = np.array(p_targets) if type(p_targets) == list else p_targets
            p_targets = torch.tensor(p_targets, dtype=torch.float32).unsqueeze(0).to(device)
            print("p_targets", p_targets.shape)
        if e_targets is not None:
            e_targets = np.array(e_targets) if type(e_targets) == list else e_targets
            e_targets = torch.tensor(e_targets, dtype=torch.float32).unsqueeze(0).to(device)
            print("e_targets", e_targets.shape)
        print("### units", units.shape, "d_targets", d_targets.shape)
        output = fs2model(speakers, units, src_lens, max_src_len) #d_targets=d_targets, p_targets=p_targets, e_targets=e_targets
    mel_len = output[9][0].item() # mel_lens
    mel_prediction = output[1][0, :mel_len].detach().transpose(0, 1) # postnet_output
    
    with torch.no_grad(): # predict Wavs from MELs
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    return wav_prediction, output

def wav2units(wav, encoder, layer=None, device='cuda'):
    ''' 
        encoder: HuBERT
    '''
    if type(wav) == np.ndarray:
        wav = torch.tensor([wav], dtype=torch.float32, device=device)
    else:
        wav = wav.to(device)
    assert type(wav) == torch.Tensor
    if len(wav.shape) == 2:
        wav = wav.unsqueeze(0)
    #print("#wav2units: ", wav.dtype, wav.shape, min(wav), max(wav))
    with torch.no_grad():  # wav -> HuBERT soft units
        if layer is None or layer < 0:
            #print("#encoder", type(encoder), "device", (next(encoder.parameters())).device)
            #print("#WAV", type(wav), wav.device)
            units = encoder.units(wav) 
        else:
            wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
            units, _ = encoder.encode(wav, layer=layer)
            
    #print("Units", units.shape)
    
    return units


def load_hubert_soft(checkpoint_path=None, rank=0, device='cuda'):
    print("### load_hubert_soft", checkpoint_path, device)
    assert checkpoint_path is not None
    print("### loading checkpoint from: ", checkpoint_path)
    if device != 'cuda':  # cpu or mps
        hubert = HubertSoft().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        hubert = HubertSoft().to(rank)
        checkpoint = torch.load(checkpoint_path, map_location={"cuda:0": f"cuda:{rank}"})

    checkpoint = checkpoint['hubert'] if checkpoint['hubert'] is not None else checkpoint
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")

    #hubert.load_state_dict(torch.load("hubert_dict.pt"), strict=True)
    hubert.load_state_dict(checkpoint, strict=True)
    hubert.eval().to(device)

    #torch.save(hubert.state_dict(), "hubert_dict.pt")
    return hubert


def my_get_vocoder(config, checkpoint_path="./hifigan/g_00205000", device='cuda'):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    assert speaker == 'universal'
    assert name == "HiFi-GAN16k"

    print("#### HiFI-GAN16k", name, speaker, device)
    with open("./hifigan/my_config_v1_16000.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    print("### HiFI-GAN ckpt", checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)

    vocoder.load_state_dict(ckpt['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder

import os

class MyWhisper2Normal(object):
    def __init__(self, args, load_encoder=True, load_decoder=True, load_vocoder=True, root=None):
        if root is None:
            print("## Lib Local Path", __file__)
            root = os.path.dirname(__file__)
        print("root", root)
        self.root = root

        self.device = device = args.device
        # set FastSpeech direct defined device
        utils.tools.device = device
        model.modules.device = device

        self.hubert = args.hubert # HuBert
        self.fastspeech2 = args.fastspeech2 # HuBert
        self.hifigan = args.hifigan
        self.args = args

        print("MyWhisper2Normal:args", args)

        # Read Config
        self.preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
        self.model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
        self.configs = (self.preprocess_config, self.model_config)
        
        if load_encoder:
            print("#### loading HuBERT")
            self.encoder = load_hubert_soft(args.hubert, device=device) # device 
            print("### HuBERT model", type(self.encoder))
        
        # FastSpeech2 HiFi-GAN
        if load_decoder:
            print("#### loading FastSpeech2")
            self.fs2model = load_fastspeech2(self.configs, checkpoint_path=args.fastspeech2, device=device) # load FastSpeech2

        if load_vocoder:
            print("### loading HiFI GAN")
            self.vocoder = my_get_vocoder(self.model_config, checkpoint_path=args.hifigan, device=device).eval()
        print("#### Done.")

    def test(self, wavfile='sample_whisper.wav', outfile='/tmp/out.wav'):
        print("### Convesion test")
        wav, sr = librosa.load(wavfile, sr=16000)
        wav_to, mel = self.convert(wav)
        print("### test:converted", type(wav_to), wav_to.shape)
        #wav_to = torch.tensor(wav_to).unsqueeze(0)
        sf.write(outfile, wav_to, 16000)
        print("### test saved to ", outfile)
        return wav_to, mel


    def whisper2normal(self, wav_from):
        if type(wav_from) != torch.Tensor:
            #wav_t = torch.tensor([wav_from], dtype=torch.float32).to(self.device)  # [1, LEN]
            wav_t = torch.tensor(wav_from, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, LEN]
        else:
            wav_t = wav_from.to(self.device)
        #print("#w2normal WAV", self.device, type(wav_t), wav_t.shape, "maxmin", wav_t.max(), wav_t.min())
        units = wav2units(wav_t, self.encoder, device=self.device) # self.device // cpu
        wav_to, _ = units2wav(units, self.fs2model, self.vocoder, self.model_config, self.preprocess_config, device=self.device)
        #print("#w2normal UNITS", units.shape, "WAV_TO", wav_to.dtype, wav_to.shape, wav_to.min(), wav_to.max())
        return wav_to, None
    
    def wav2units(self, wav_t):
        return wav2units(wav_t, self.encoder, device=self.device)
    
    def units2wav(self, units, p_targets=None, e_targets=None):
        return units2wav(units, self.fs2model, self.vocoder, self.model_config, self.preprocess_config, p_targets=p_targets, e_targets=e_targets, device=self.device)

    def convert(self, wav_from):
        wav_to, mel = self.whisper2normal(wav_from)
        return wav_to, mel


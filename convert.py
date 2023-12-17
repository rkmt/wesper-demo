import numpy as np
import os
import torch
import librosa
import soundfile as sf

from whisper_normal import MyWhisper2Normal

'''
Whisper to Normal conversion demo
'''

def main(args):
    w2n = MyWhisper2Normal(args)
    #w2n.test()
    whisp_wav, sr = librosa.load(args.input, sr=16000)
    print("Loaed wav", args.input,  "len", len(whisp_wav))
    normal_wav, _ = w2n.convert(whisp_wav)
    print("Converted wav", args.output,  "len", len(normal_wav))
    sf.write(args.output, normal_wav, 16000)
          

import argparse

def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
#    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
#        return "mps" # M1 mac GPU
    else:
        return "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        default = "sample_whisper.wav"
    )

    parser.add_argument("--output",
                        default = "converted.wav"
    )

    parser.add_argument("--preprocess_config",
        default = 'config/my_preprocess16k_LJ.yaml'
    )

    parser.add_argument("--model_config",
        default = 'config/my_model16000.yaml'
    )

    '''
    parser.add_argument("--train_config",
        default = 'config/my_train16k_LJ.yaml'
    )
    '''

    parser.add_argument("--device", 
        default = get_default_device()
    )

    parser.add_argument("--hubert", 
        help="hubert checkpoint path",
        #default="models/hubert/model-layer12-450000.pt"
        default="https://github.com/rkmt/wesper-demo/releases/download/v0.1/model-layer12-450000.pt",

    )

    parser.add_argument("--fastspeech2", 
        help="fastspeech2 checkpoint path",
        #default="models/fastspeech2/lambda_best.tar"
        #default="models/fastspeech2/googletts_neutral_best.tar"        
        default="https://github.com/rkmt/wesper-demo/releases/download/v0.1/googletts_neutral_best.tar",
    )

    parser.add_argument("--hifigan", 
        help="hifigan checkpoint path",
        #default="./hifigan/g_00205000"
        default="https://github.com/rkmt/wesper-demo/releases/download/v0.1/g_00205000",
    )

    args = parser.parse_args()   

    print("### args ###\n", args)
    main(args)
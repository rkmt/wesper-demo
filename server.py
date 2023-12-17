'''
WESPER server (whisper to normal)
'''

import zmq
import numpy as np
import os
import torch
import librosa
import soundfile as sf

from whisper_normal import MyWhisper2Normal

# Audio recording parameters
RATE = 16000
PORT = 5557

class DummyMyWhisper2Normal(object):
    def __init__(self, args):
        print("Dummy W2n")

    def convert(self, audio_from):
        return audio_from, None


class ZmqLoop(object):
    def __init__(self, w2n, port=PORT):
        print(f"\n### create zmq port={port}")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.w2n = w2n
        print("\n### server starts")

    def mainloop(self):
        while True:
            print("### waiting..")
            data = None
            try:
                data = self.socket.recv_pyobj()
            except Exception as e:
                print("Exception socket error:", e)
                continue

            if data is not None: # whipser recognition
                whisp_wav = data
                normal_wav, _ = self.w2n.convert(whisp_wav)
                print(f"### converted data:{len(data)} whisp_wav:{len(whisp_wav)} normal_wav:{len(normal_wav)}")

            self.socket.send_pyobj((normal_wav,))

    def destroy(self):
        self.socket.close()
        self.context.destroy()


def main(args):
    w2n = MyWhisper2Normal(args)
    #w2n = DummyMyWhisper2Normal(args)

    zmq_loop = ZmqLoop(w2n, port=args.port)
    zmq_loop.mainloop()

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

    parser.add_argument("--preprocess_config",
        #default = '../../gitdownloads/FastSpeech2/config/LJSpeech/my_preprocess16000.yaml'
        #default = 'config/LJSpeech/my_preprocess.yaml'
        #default = 'config/LJSpeech/my_preprocess16k_LJ.yaml'
        default = 'config/my_preprocess16k_LJ.yaml'
    )

    parser.add_argument("--model_config",
        #args.model_config = 'config/LJSpeech/model.yaml'
        #args.model_config = 'config/LJSpeech/my_model.yaml'
        #default = '../../gitdownloads/FastSpeech2/config/LJSpeech/my_model16000.yaml'
        default = 'config/my_model16000.yaml'
    )

    parser.add_argument("--train_config",
        #default = '../../gitdownloads/FastSpeech2/config/LJSpeech/my_train16000.yaml'
        #args.train_config = 'config/LJSpeech/my_train.yaml'
        #args.train_config = 'config/LJSpeech/my_train16k_LJ.yaml'
        default = 'config/my_train16k_LJ.yaml'
    )

    parser.add_argument("--device", 
        #default= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        default = get_default_device()
    )

    parser.add_argument("--port", type=int, default=PORT, help="server port number")

    parser.add_argument("--hubert", 
        help="hubert checkpoint path",
        default="models/hubert/model-layer12-450000.pt"
    )

    parser.add_argument("--fastspeech2", 
        help="fastspeech2 checkpoint path",
        #default="models/fastspeech2/lambda_best.tar"
        default="models/fastspeech2/googletts_neutral_best.tar"        
    )

    parser.add_argument("--hifigan", 
        help="hifigan checkpoint path",
        default="./hifigan/g_00205000"
    )

    args = parser.parse_args()   

    print("### args", args)
    main(args)

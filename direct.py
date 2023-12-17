
import numpy as np
import os
import torch
import librosa
import soundfile as sf
import time
import datetime
import subprocess
import pyaudio
import soundfile as sf
import sounddevice as sd


from whisper_normal import MyWhisper2Normal

# Audio recording parameters
RATE = 16000

class MySpeaker():
    def __init__(self, pa=None):
        if pa is None:
            pa = pyaudio.PyAudio()
        self.pa = pa

    def play(self, wav):
        print("play ", len(wav))
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            output=True)
        chunk = 1024
        for i in range(0, len(wav), chunk):
            data = wav[i:min(i+chunk, len(wav)-i)]
            self.stream.write(data)
        self.stream.close()
        self.stream = None

    def destroy(self):
        self.stream.close()
        self.pa.terminate()
        self.stream = None
        self.pa = None

class MyAudioClientDirect(object):
    def __init__(self, args, sr=RATE):
        self.args = args
        self.sr = sr
        self.save_root = "convlog"
        self.w2n = MyWhisper2Normal(args)
        #self.speaker = MySpeaker(pa)

    def destroy(self):
        print("#### destroy called")

    def call(self, audio):
        #print("CALL ", len(audio))
        #sf.write("record.wav", audio, RATE)
        #subprocess.run(["afplay","record.wav"])
        print("### SEND: ", type(audio), len(audio))

        start_time = time.time()
        normal_wav, _ = self.w2n.convert(audio)
        process_time = (time.time() - start_time)

        print("### converted --->", len(audio), len(normal_wav))
        sec = (len(normal_wav) / RATE) 
        print("### RES", sec, "process_time", process_time, "x", sec / process_time)

        #self.speaker.play(normal_wav)

        # test playback
        sd.play(normal_wav, self.sr)
        sd.wait()

        '''
        sf.write("res.wav", normal_wav, RATE)
        subprocess.run(["afplay","res.wav"])
        '''

        '''
        # record converted samples
        now = datetime.datetime.now()
        fname = now.strftime('%y%m%d-%H%M%S')
        sf.write(os.path.join(self.save_root, f"{fname}-from.wav"), audio, 16000)
        sf.write(os.path.join(self.save_root, f"{fname}-to.wav"), normal_wav, 16000)        
        '''


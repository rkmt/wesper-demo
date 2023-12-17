from __future__ import division

import argparse
import re
import sys
import os
import pyaudio
import numpy as np
import tkinter as tk
import time
import zmq
import soundfile as sf
import subprocess
import datetime

# Audio recording parameters
RATE = 16000
FORMAT = pyaudio.paInt16
CHUNK=1024
CHANNELS=1
PORT=5557

class MyAudioClient(object):
    def __init__(self, host=None, port=PORT):
        assert host is not None
        server_address=f"tcp://{host}:{port}"
        print("### connection", server_address)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(server_address)
        self.save_root = "convlog"

    def destroy(self):
        self.socket.close()
        self.context.destroy()

    def call(self, audio):
        #print("CALL ", len(audio))
        sf.write("record.wav", audio, 16000)
        #subprocess.run(["afplay","record.wav"])
        print("### SEND: ", type(audio), len(audio))
        start_time = time.time()
        self.socket.send_pyobj(audio)
        res = self.socket.recv_pyobj()  # [id, is_whisper, is_normal, rmx, trans_list]
        process_time = (time.time() - start_time)
        sec = (len(res[0]) / 16000) 
        assert res is not None
        print("### RES", len(res), len(res[0]), sec, process_time, sec / process_time)
        # test playback
        sf.write("res.wav", res[0], 16000)
        subprocess.run(["afplay","res.wav"])

        '''
        # record converted samples
        now = datetime.datetime.now()
        fname = now.strftime('%y%m%d-%H%M%S')
        sf.write(os.path.join(self.save_root, f"{fname}-from.wav"), audio, 16000)
        sf.write(os.path.join(self.save_root, f"{fname}-to.wav"), res[0], 16000)        
        '''



class Microphone(object):
    """ 押している間だけ録音するマイク """

    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = []
        self.closed = True
        self._audio_stream = None
        self._audio_interface = pyaudio.PyAudio()

    def start_recording(self):
        self._buff = []
        assert self._audio_stream is None
        #assert self._audio_interface is None
        #self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        print("start recording")
        return self

    def stop_recording(self):
        assert not self.closed and self._audio_stream is not None
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._audio_stream = None
        #self._audio_interface.terminate()
        #self._audio_interface = None
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        data = b''.join(self._buff)
        print("data len", len(data))
        audio = np.frombuffer(data, dtype="int16") / 32768.0
        self._buff = []
        print("stop recording")
        return audio

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.append(in_data)
        return None, pyaudio.paContinue


class MyGUI(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.button = tk.Button(self, text="Record", width=40, height=2)
        self.text = tk.Text(self, width=40, height=10)
        self.vsb = tk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=self.vsb.set)

        self.button.pack(side="top")
        self.vsb.pack(side="right", fill="y")
        self.text.pack(side="bottom", fill="x")

        self.button.bind("<ButtonPress>", self.on_press)
        self.button.bind("<ButtonRelease>", self.on_release)

        self.mic = Microphone()

    def set_client(self, host=None):
        self.client = MyAudioClient(host=host)

    def on_press(self, event):
        self.log("button was pressed")
        self.mic.start_recording()
        self.log("start")

    def on_release(self, event):
        self.log("button was released")
        audio = self.mic.stop_recording()
        print(len(audio))
        self.log("call")
        self.client.call(audio)
        self.log("get")

    def log(self, message):
        now = time.strftime("%I:%M:%S", time.localtime())
        self.text.insert("end", now + " " + message.strip() + "\n")
        self.text.see("end")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--server",
        default = 'rkmta101'
    )

    args = parser.parse_args()   
    host = args.server
    print("### args", host, args)

    root = tk.Tk()
    mygui = MyGUI(root)
    mygui.pack(side="top", fill="both", expand=True)
    mygui.set_client(host)
    root.mainloop()


import pyaudio
import wave
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import os
import tempfile
from typing import Optional, Tuple
import threading
import time

class AudioTools:
    """音频工具类，提供麦克风录制和扬声器播放功能"""
    
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.recording_thread = None
        
    def record_audio(self, duration: float = 5.0) -> str:
        """
        录制音频
        
        Args:
            duration: 录制时长（秒）
            
        Returns:
            录制的音频文件路径
        """
        try:
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_filename = temp_file.name
            temp_file.close()
            
            # 录制音频
            frames = []
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print(f"开始录制音频，时长: {duration}秒...")
            
            for i in range(0, int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)
                
            print("录制完成")
            
            stream.stop_stream()
            stream.close()
            
            # 保存为WAV文件
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            return temp_filename
            
        except Exception as e:
            print(f"录制音频时出错: {e}")
            return None
    
    def play_audio(self, audio_file: str) -> bool:
        """
        播放音频文件
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            是否播放成功
        """
        try:
            if not os.path.exists(audio_file):
                print(f"音频文件不存在: {audio_file}")
                return False
            
            # 读取音频文件
            sample_rate, data = wav.read(audio_file)
            
            print(f"开始播放音频: {audio_file}")
            
            # 播放音频
            sd.play(data, sample_rate)
            sd.wait()  # 等待播放完成
            
            print("音频播放完成")
            return True
            
        except Exception as e:
            print(f"播放音频时出错: {e}")
            return False
    
    def text_to_speech(self, text: str, output_file: Optional[str] = None) -> str:
        """
        文本转语音
        
        Args:
            text: 要转换的文本
            output_file: 输出文件路径（可选）
            
        Returns:
            生成的音频文件路径
        """
        try:
            import pyttsx3
            
            # 初始化TTS引擎
            engine = pyttsx3.init()
            
            # 设置语音属性
            engine.setProperty('rate', 150)  # 语速
            engine.setProperty('volume', 0.9)  # 音量
            
            # 如果没有指定输出文件，创建临时文件
            if output_file is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                output_file = temp_file.name
                temp_file.close()
            
            # 生成语音
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            
            print(f"文本转语音完成: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"文本转语音时出错: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'audio'):
            self.audio.terminate()

# 全局音频工具实例
audio_tools = AudioTools() 
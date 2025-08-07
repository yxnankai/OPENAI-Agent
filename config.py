import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """应用配置类"""
    
    # OpenAI配置
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    
    # 音频配置
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    AUDIO_CHUNK_SIZE = 1024
    
    # 应用配置
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("请设置OPENAI_API_KEY环境变量")
        return True 
#!/usr/bin/env python3
"""
主启动脚本 - 自动启动Flask服务并打开浏览器
"""
import os
import sys
import time
import threading
import webbrowser
import logging
from datetime import datetime
from app_flask import app
from config import Config

# 配置日志
def setup_logging():
    """设置日志配置"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"ai_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台（如果有的话）
        ]
    )
    return logging.getLogger(__name__)

def open_browser():
    """延迟打开浏览器"""
    time.sleep(2)  # 等待Flask服务启动
    try:
        webbrowser.open('http://localhost:5000')
        logger.info("🌐 浏览器已自动打开: http://localhost:5000")
    except Exception as e:
        logger.error(f"❌ 自动打开浏览器失败: {e}")
        logger.info("请手动打开浏览器访问: http://localhost:5000")

def main():
    """主函数"""
    global logger
    logger = setup_logging()
    
    logger.info("🚀 正在启动AI Agent应用...")
    logger.info(f"📋 配置信息:")
    logger.info(f"   - OpenAI模型: {Config.OPENAI_MODEL}")
    logger.info(f"   - API地址: {Config.OPENAI_BASE_URL}")
    logger.info(f"   - 端口: 5000")
    logger.info("")
    
    # 检查配置
    if not Config.OPENAI_API_KEY:
        logger.warning("⚠️  警告: 未设置OPENAI_API_KEY环境变量")
        logger.info("请在.env文件中设置您的OpenAI API密钥")
        logger.info("")
    
    # 在新线程中打开浏览器
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # 启动Flask应用
    logger.info("🌍 启动Web服务...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("👋 应用已停止")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main() 
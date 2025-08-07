#!/usr/bin/env python3
"""
主启动脚本 - 带系统托盘的后台运行版本
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

# 尝试导入系统托盘相关库
try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    print("提示: 安装 pystray 和 pillow 可以获得系统托盘功能")
    print("pip install pystray pillow")

# 全局变量
logger = None
tray_icon = None
server_thread = None
app_running = True

def create_tray_icon():
    """创建系统托盘图标"""
    if not TRAY_AVAILABLE:
        return None
    
    # 创建一个简单的图标
    def create_icon():
        # 创建一个32x32的图像
        image = Image.new('RGB', (32, 32), color='white')
        draw = ImageDraw.Draw(image)
        # 画一个简单的AI图标
        draw.ellipse([8, 8, 24, 24], outline='blue', width=2)
        draw.text((12, 12), "AI", fill='blue')
        return image
    
    def on_clicked(icon, item):
        if str(item) == "打开浏览器":
            webbrowser.open('http://localhost:5000')
        elif str(item) == "查看日志":
            log_dir = "logs"
            if os.path.exists(log_dir):
                os.startfile(log_dir)
        elif str(item) == "退出":
            global app_running
            app_running = False
            icon.stop()
            os._exit(0)
    
    # 创建菜单
    menu = pystray.Menu(
        pystray.MenuItem("打开浏览器", on_clicked),
        pystray.MenuItem("查看日志", on_clicked),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", on_clicked)
    )
    
    # 创建托盘图标
    icon = pystray.Icon("AI Agent", create_icon(), "AI Agent 正在运行", menu)
    return icon

def run_flask_app():
    """在后台运行Flask应用"""
    global logger
    try:
        logger.info("🌍 启动Web服务...")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"❌ Flask服务启动失败: {e}")

def open_browser():
    """延迟打开浏览器"""
    time.sleep(3)  # 等待Flask服务启动
    try:
        webbrowser.open('http://localhost:5000')
        logger.info("🌐 浏览器已自动打开: http://localhost:5000")
    except Exception as e:
        logger.error(f"❌ 自动打开浏览器失败: {e}")
        logger.info("请手动打开浏览器访问: http://localhost:5000")

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

def main():
    """主函数"""
    global logger, tray_icon, server_thread
    
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
    
    # 创建系统托盘图标
    if TRAY_AVAILABLE:
        tray_icon = create_tray_icon()
        logger.info("📌 系统托盘图标已创建")
    
    # 在新线程中启动Flask服务
    server_thread = threading.Thread(target=run_flask_app, daemon=True)
    server_thread.start()
    
    # 在新线程中打开浏览器
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # 运行系统托盘
    if TRAY_AVAILABLE and tray_icon:
        logger.info("🎯 应用已在后台运行，请查看系统托盘")
        tray_icon.run()
    else:
        # 如果没有系统托盘，则在前台运行
        logger.info("🎯 应用正在运行...")
        try:
            while app_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("👋 应用已停止")

if __name__ == "__main__":
    main() 
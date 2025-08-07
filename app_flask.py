#!/usr/bin/env python3
"""
AI Agent Flask Web应用
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import tempfile
# from models import get_available_models, update_models_from_api  # 已删除models.py文件
from config import Config
from langchain_agent import unified_agent

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 配置
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局变量存储对话历史
conversation_history = []

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """AI对话API - 使用统一的LangChain Agent"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': '消息不能为空'}), 400
        
        # 使用统一的LangChain Agent（带工具调用记录）
        result = unified_agent.chat_with_tool_calls(message)
        
        if not result['success']:
            return jsonify({'error': result['response']}), 500
        
        response = result['response']
        tool_calls = result['tool_calls']
        
        # 添加到对话历史
        conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': request.json.get('timestamp', ''),
            'agent_type': 'unified_langchain'
        })
        conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': request.json.get('timestamp', ''),
            'agent_type': 'unified_langchain',
            'tool_calls': tool_calls
        })
        
        return jsonify({
            'response': response,
            'success': True,
            'agent_type': 'unified_langchain',
            'tool_calls': tool_calls
        })
        
    except Exception as e:
        return jsonify({'error': f'对话出错: {str(e)}'}), 500

# 所有功能已集成到统一的LangChain Agent中，不再需要单独的API端点

@app.route('/api/reset_conversation', methods=['POST'])
def reset_conversation():
    """重置对话历史API"""
    try:
        global conversation_history
        conversation_history.clear()
        
        return jsonify({
            'success': True,
            'message': '对话历史已重置'
        })
        
    except Exception as e:
        return jsonify({'error': f'重置出错: {str(e)}'}), 500

@app.route('/api/conversation_history', methods=['GET'])
def get_conversation_history():
    """获取对话历史API"""
    try:
        return jsonify({
            'success': True,
            'history': conversation_history
        })
        
    except Exception as e:
        return jsonify({'error': f'获取历史出错: {str(e)}'}), 500

@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """提供图片文件服务"""
    try:
        import os
        from flask import send_file, abort
        
        # 解码URL编码的路径
        import urllib.parse
        decoded_path = urllib.parse.unquote(image_path)
        
        # 安全检查：确保路径是相对路径或允许的绝对路径
        if os.path.isabs(decoded_path):
            # 允许项目目录下的文件和临时目录下的文件
            project_dir = os.path.abspath(os.path.dirname(__file__))
            temp_dir = tempfile.gettempdir()
            
            # 检查是否在项目目录或临时目录中
            if not (decoded_path.startswith(project_dir) or decoded_path.startswith(temp_dir)):
                return jsonify({'error': '访问被拒绝'}), 403
        else:
            # 对于相对路径，转换为绝对路径
            project_dir = os.path.abspath(os.path.dirname(__file__))
            decoded_path = os.path.join(project_dir, decoded_path)
        
        # 检查文件是否存在
        if not os.path.exists(decoded_path):
            return jsonify({'error': f'文件不存在: {decoded_path}'}), 404
        
        # 检查文件扩展名
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        file_ext = os.path.splitext(decoded_path)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': '不支持的文件类型'}), 400
        
        # 发送文件
        return send_file(decoded_path, mimetype='image/*')
        
    except Exception as e:
        return jsonify({'error': f'图片服务错误: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """系统状态API"""
    try:
        tools = unified_agent.get_available_tools()
        tool_names = [tool['name'] for tool in tools]
        
        return jsonify({
            'success': True,
            'model': Config.OPENAI_MODEL,
            'model_name': '统一LangChain Agent',
            'model_description': '集成所有功能的统一AI Agent',
            'api_url': Config.OPENAI_BASE_URL,
            'conversation_count': len(conversation_history) // 2,
            'tools_available': tool_names
        })
        
    except Exception as e:
        return jsonify({'error': f'状态查询出错: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    try:
        # 提供默认的模型列表
        models = [
            {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo', 'description': '最新的GPT-4模型'},
            {'id': 'gpt-4', 'name': 'GPT-4', 'description': 'GPT-4模型'},
            {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo', 'description': 'GPT-3.5 Turbo模型'},
            {'id': 'gpt-4o', 'name': 'GPT-4o', 'description': 'GPT-4o模型'},
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini', 'description': 'GPT-4o Mini模型'}
        ]
        return jsonify({
            'success': True,
            'models': models,
            'current_model': Config.OPENAI_MODEL
        })
    except Exception as e:
        return jsonify({'error': f'获取模型列表失败: {str(e)}'}), 500

@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """切换模型"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({'error': '请提供模型ID'}), 400
        
        # 更新配置中的模型
        Config.OPENAI_MODEL = model_id
        return jsonify({
            'success': True,
            'message': f'成功切换到模型: {model_id}',
            'current_model': Config.OPENAI_MODEL
        })
            
    except Exception as e:
        return jsonify({'error': f'切换模型失败: {str(e)}'}), 500

@app.route('/api/models/update', methods=['POST'])
def update_models():
    """从API更新模型列表"""
    try:
        # 由于删除了models.py，这里返回默认模型列表
        models = [
            {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo', 'description': '最新的GPT-4模型'},
            {'id': 'gpt-4', 'name': 'GPT-4', 'description': 'GPT-4模型'},
            {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo', 'description': 'GPT-3.5 Turbo模型'},
            {'id': 'gpt-4o', 'name': 'GPT-4o', 'description': 'GPT-4o模型'},
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini', 'description': 'GPT-4o Mini模型'}
        ]
        return jsonify({
            'success': True,
            'message': '模型列表已更新',
            'models': models
        })
            
    except Exception as e:
        return jsonify({'error': f'更新模型列表失败: {str(e)}'}), 500

@app.route('/api/langchain/tools', methods=['GET'])
def get_langchain_tools():
    """获取统一LangChain Agent可用工具列表"""
    try:
        tools = unified_agent.get_available_tools()
        return jsonify({
            'success': True,
            'tools': tools
        })
    except Exception as e:
        return jsonify({'error': f'获取工具列表失败: {str(e)}'}), 500

@app.route('/api/text_to_speech', methods=['POST'])
def text_to_speech():
    """文本转语音API"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': '文本内容不能为空'
            }), 400
        
        # 直接使用文本转语音工具，而不是通过Agent
        from langchain_agent import TextToSpeechTool
        tts_tool = TextToSpeechTool()
        response = tts_tool._run(text)
        
        # 检查是否成功
        if "✅ 文本转语音成功" in response:
            return jsonify({
                'success': True,
                'message': '语音播放成功',
                'response': response
            })
        else:
            return jsonify({
                'success': False,
                'error': response
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tool_call_info', methods=['POST'])
def get_tool_call_info():
    """获取工具调用信息"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'success': False,
                'error': '消息内容不能为空'
            }), 400
        
        # 获取工具调用信息
        tool_info = unified_agent.get_tool_call_info(message)
        
        return jsonify({
            'success': True,
            'tool_info': tool_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """下载文件API"""
    try:
        # 安全检查：只允许下载临时目录中的文件
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        if os.path.exists(file_path) and file_path.startswith(temp_dir):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在或访问被拒绝'}), 404
            
    except Exception as e:
        return jsonify({'error': f'下载出错: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '接口不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    # 检查环境变量
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ 错误: 请设置 OPENAI_API_KEY 环境变量")
        print("💡 提示: 请在项目根目录创建 .env 文件并添加您的 OpenAI API 密钥")
        exit(1)
    
    print("🚀 启动AI Agent Flask应用...")
    print("📱 访问地址: http://localhost:5000")
    print("🔧 API文档: http://localhost:5000/api/status")
    
    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    ) 
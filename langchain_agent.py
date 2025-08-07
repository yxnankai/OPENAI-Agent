#!/usr/bin/env python3
"""
基于LangChain的统一AI Agent
集成所有功能：语音、网络查询、本地工具等
"""
import os
import json
import tempfile
import wave
import numpy as np
from typing import List, Dict, Any, Optional, ClassVar, Type, Type
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import webbrowser
import subprocess
import platform
import requests
from bs4 import BeautifulSoup
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wav
from config import Config
from tools.web_search import enhanced_search
from tools.document_reader import DocumentReader
from tools.camera_tools import camera_manager
from tools.local_models import local_model_manager

# 全局工具调用记录器
tool_recorder = {
    'tool_calls': []
}

def record_tool_call(tool_name: str, tool_input: dict, tool_output: str):
    """记录工具调用"""
    tool_recorder['tool_calls'].append({
        'tool': tool_name,
        'input': tool_input,
        'output': tool_output,
        'status': 'completed'
    })

def get_tool_calls():
    """获取工具调用记录"""
    return tool_recorder['tool_calls'].copy()

def clear_tool_calls():
    """清空工具调用记录"""
    tool_recorder['tool_calls'].clear()

# 为所有工具类添加记录功能的基础类
class RecordableTool(BaseTool):
    """可记录调用的工具基类"""
    
    def _run(self, *args, **kwargs):
        """重写_run方法以添加记录功能"""
        try:
            # 记录输入
            tool_input = kwargs if kwargs else args[0] if args else {}
            
            # 执行原始方法
            result = self._original_run(*args, **kwargs)
            
            # 记录输出
            record_tool_call(self.name, tool_input, result)
            
            return result
        except Exception as e:
            error_msg = f"❌ 工具执行错误: {str(e)}"
            record_tool_call(self.name, kwargs if kwargs else args, error_msg)
            raise e
    
    def _original_run(self, *args, **kwargs):
        """原始的运行方法，由子类实现"""
        raise NotImplementedError("子类必须实现_original_run方法")

class AudioRecordTool(RecordableTool):
    """音频录制工具"""
    name: str = "audio_record"
    description: str = "录制音频，参数为录制时长（秒）"
    
    class InputSchema(BaseModel):
        duration: float = Field(description="录制时长（秒）")
    
    def _original_run(self, duration: float, **kwargs) -> str:
        """录制音频"""
        try:
            # 录制音频
            sample_rate = 16000
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            
            # 保存音频文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            wav.write(temp_file.name, sample_rate, recording)
            
            return f"✅ 音频录制成功，时长: {duration}秒，文件: {temp_file.name}"
        except Exception as e:
            return f"❌ 音频录制失败: {str(e)}"

class AudioPlayTool(RecordableTool):
    """音频播放工具"""
    name: str = "audio_play"
    description: str = "播放音频文件"
    
    class InputSchema(BaseModel):
        audio_file: str = Field(description="音频文件路径")
    
    def _original_run(self, audio_file: str, **kwargs) -> str:
        """播放音频"""
        try:
            if not os.path.exists(audio_file):
                return f"❌ 音频文件不存在: {audio_file}"
            
            # 播放音频
            sample_rate, data = wav.read(audio_file)
            sd.play(data, sample_rate)
            sd.wait()
            
            return f"✅ 音频播放完成: {audio_file}"
        except Exception as e:
            return f"❌ 音频播放失败: {str(e)}"

class TextToSpeechTool(RecordableTool):
    """文本转语音工具"""
    name: str = "text_to_speech"
    description: str = "将文本转换为语音并播放"
    
    class InputSchema(BaseModel):
        text: str = Field(description="要转换的文本")
    
    def _original_run(self, text: str, **kwargs) -> str:
        """文本转语音"""
        try:
            # 初始化TTS引擎
            engine = pyttsx3.init()
            
            # 设置语音属性
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            # 直接播放，不保存文件
            engine.say(text)
            engine.runAndWait()
            
            return f"✅ 文本转语音成功并播放: {text}"
        except Exception as e:
            return f"❌ 文本转语音失败: {str(e)}"

class WebSearchTool(RecordableTool):
    """网页搜索工具"""
    name: str = "web_search"
    description: str = "搜索网络信息"
    
    class InputSchema(BaseModel):
        query: str = Field(description="搜索关键词")
        max_results: int = Field(default=5, description="最大结果数量")
    
    def _original_run(self, query: str, max_results: int = 5, **kwargs) -> str:
        """网页搜索"""
        try:
            # 使用增强的搜索工具
            return enhanced_search.search_and_format(query, max_results)
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}\n\n建议您直接使用浏览器进行搜索。"

class WebSummaryTool(RecordableTool):
    """网页搜索并总结工具"""
    name: str = "web_summary"
    description: str = "搜索网络信息并提供总结"
    
    class InputSchema(BaseModel):
        query: str = Field(description="搜索关键词")
    
    def _original_run(self, query: str, **kwargs) -> str:
        """搜索并总结"""
        try:
            # 先进行搜索
            search_result = self._search_web(query)
            
            # 返回详细的搜索结果，并要求AI进行总结
            if "未找到相关结果" not in search_result and "搜索失败" not in search_result:
                return f"""🔍 搜索结果: {query}

{search_result}

📋 请基于以上搜索结果，提供详细的信息总结和分析：

1. **主要信息摘要**：提取核心事实和数据
2. **信息来源分析**：评估各来源的权威性和时效性  
3. **关键发现**：总结重要趋势和观点
4. **参考资料**：明确标注每个信息片段对应的来源链接

请确保在总结中明确引用具体的链接，格式如：[来源标题](链接URL)，并提供客观、准确的分析。"""
            else:
                return f"🔍 搜索结果: {query}\n\n{search_result}\n\n💡 建议：如果搜索结果不理想，您可以尝试使用不同的关键词或直接使用浏览器搜索。"
            
        except Exception as e:
            return f"❌ 搜索总结失败: {str(e)}"
    
    def _search_web(self, query: str) -> str:
        """内部搜索方法"""
        try:
            # 使用增强的搜索工具
            return enhanced_search.search_and_format(query, 5)
        except Exception as e:
            return f"搜索失败: {str(e)}，建议直接使用浏览器搜索。"

class BrowserTool(RecordableTool):
    """浏览器工具 - 打开网页"""
    name: str = "browser_open"
    description: str = "打开指定URL的网页浏览器"
    
    class InputSchema(BaseModel):
        url: str = Field(description="要打开的网页URL")
    
    def _original_run(self, url: str, **kwargs) -> str:
        """打开网页"""
        try:
            webbrowser.open(url)
            return f"✅ 成功打开网页: {url}"
        except Exception as e:
            return f"❌ 打开网页失败: {str(e)}"

class SystemCommandTool(RecordableTool):
    """系统命令工具 - 执行本地命令和搜索"""
    name: str = "system_command"
    description: str = "在本地系统执行命令，包括搜索、文件操作、网络查询等"
    
    # 类变量，用于记录待确认的命令
    pending_commands: ClassVar[dict] = {}
    
    class InputSchema(BaseModel):
        command: str = Field(description="要执行的系统命令，如：dir、ping、curl、find等")
    
    def _original_run(self, command: str, **kwargs) -> str:
        """执行系统命令"""
        try:
            # 检查是否是重复的关机命令（确认操作）
            if 'shutdown' in command.lower():
                # 检查是否之前已经提示过这个命令
                if command in self.pending_commands:
                    # 用户确认了，执行关机命令
                    del self.pending_commands[command]  # 清除待确认状态
                    return self._execute_shutdown(command)
                else:
                    # 第一次发送关机命令，要求确认
                    self.pending_commands[command] = True
                    return f"⚠️ 即将执行关机命令: {command}\n\n请确认您真的要关闭计算机吗？\n\n如果确认，请再次发送相同的命令。"
            
            # 安全检查：禁止执行危险命令（但允许关机命令）
            dangerous_commands = ['format', 'del /s', 'rm -rf', 'taskkill /f']
            if any(dangerous in command.lower() for dangerous in dangerous_commands):
                return f"❌ 安全限制：禁止执行危险命令 '{command}'"
            
            # 优化文件搜索命令
            if command.lower().startswith('dir') and '*python*' in command.lower():
                # 优化Python文件搜索 - 搜索D盘
                command = 'dir /s /b "D:\\*python*"'
            elif command.lower().startswith('dir') and '*模型*' in command.lower():
                # 优化模型文件搜索 - 搜索D盘
                command = 'dir /s /b "D:\\*模型*"'
            elif command.lower().startswith('dir') and '*model*' in command.lower():
                # 优化model文件搜索 - 搜索D盘
                command = 'dir /s /b "D:\\*model*"'
            elif command.lower().startswith('dir') and '*config*' in command.lower():
                # 优化config文件搜索 - 搜索D盘
                command = 'dir /s /b "D:\\*config*"'
            elif command.lower().startswith('dir') and '*test*' in command.lower():
                # 优化test文件搜索 - 搜索D盘
                command = 'dir /s /b "D:\\*test*"'
            
            # 执行命令
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    return f"✅ 命令执行成功:\n\n{output}"
                else:
                    return f"✅ 命令执行成功，但无输出: {command}"
            else:
                error = result.stderr.strip()
                if error:
                    return f"❌ 命令执行失败: {error}"
                else:
                    return f"❌ 命令执行失败，返回码: {result.returncode}"
                    
        except subprocess.TimeoutExpired:
            return f"❌ 命令执行超时: {command}"
        except Exception as e:
            return f"❌ 命令执行错误: {str(e)}"
    
    def _execute_shutdown(self, command: str) -> str:
        """执行关机命令"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return f"✅ 关机命令已执行: {command}\n\n计算机将在60秒后关闭。"
            else:
                return f"❌ 关机命令执行失败: {result.stderr}"
        except Exception as e:
            return f"❌ 关机命令执行错误: {str(e)}"

class FileOperationTool(RecordableTool):
    """文件操作工具"""
    name: str = "file_operation"
    description: str = "执行文件操作（创建、读取、写入文件等）"
    
    class InputSchema(BaseModel):
        operation: str = Field(description="操作类型：read, write, create, delete")
        file_path: str = Field(description="文件路径")
        content: Optional[str] = Field(description="文件内容（写入时使用）")
    
    def _original_run(self, operation: str, file_path: str = None, content: Optional[str] = None, **kwargs) -> str:
        """执行文件操作"""
        try:
            # 处理参数传递问题
            if 'file_path' in kwargs:
                file_path = kwargs['file_path']
            elif 'path' in kwargs:
                file_path = kwargs['path']
            elif file_path is None:
                # 如果还是没有，尝试从第一个位置参数获取
                args_list = list(kwargs.values())
                if args_list:
                    file_path = str(args_list[0])
            
            if not file_path:
                return "❌ 缺少文件路径"
            
            if operation == "read":
                if not os.path.exists(file_path):
                    return f"❌ 文件不存在: {file_path}"
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"✅ 文件读取成功\n\n📄 文件路径: {file_path}\n📝 内容:\n{content}"
            
            elif operation == "write":
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content or "")
                return f"✅ 文件写入成功\n\n📄 文件路径: {file_path}"
            
            elif operation == "create":
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content or "")
                return f"✅ 文件创建成功\n\n📄 文件路径: {file_path}"
            
            elif operation == "delete":
                if not os.path.exists(file_path):
                    return f"❌ 文件不存在: {file_path}"
                
                os.remove(file_path)
                return f"✅ 文件删除成功\n\n📄 文件路径: {file_path}"
            
            else:
                return f"❌ 不支持的操作类型: {operation}"
                
        except Exception as e:
            return f"❌ 文件操作失败: {str(e)}"

class CommandLineSearchTool(RecordableTool):
    """命令行搜索工具"""
    name: str = "cmd_search"
    description: str = "通过命令行进行搜索和查询，包括文件搜索、网络查询、系统信息等"
    
    class InputSchema(BaseModel):
        search_type: str = Field(description="搜索类型：file(文件搜索)、network(网络查询)、system(系统信息)、process(进程查询)")
        query: str = Field(description="搜索查询内容")
        options: Optional[str] = Field(description="搜索选项，如路径、参数等")
    
    def _original_run(self, search_type: str = None, query: str = None, options: Optional[str] = None, **kwargs) -> str:
        """执行命令行搜索"""
        try:
            # 处理参数传递问题 - 修复参数解析
            print(f"DEBUG cmd_search: 接收到的参数 - search_type={search_type}, query={query}, options={options}, kwargs={kwargs}")
            
            # 从kwargs中正确提取参数
            if 'keyword' in kwargs:
                query = kwargs['keyword']
            elif 'query' in kwargs:
                query = kwargs['query']
            elif query is None:
                # 如果还是没有，尝试从第一个位置参数获取
                args_list = list(kwargs.values())
                if args_list:
                    query = str(args_list[0])
            
            if search_type is None:
                search_type = kwargs.get('search_type', 'file')
            
            # 处理搜索路径参数
            if 'path' in kwargs:
                options = kwargs['path']
            elif 'search_path' in kwargs:
                options = kwargs['search_path']
            elif 'options' in kwargs:
                options = kwargs['options']
            
            print(f"DEBUG cmd_search: 解析后的参数 - search_type={search_type}, query={query}, options={options}")
            
            if not query:
                return "❌ 缺少搜索查询内容"
            
            # 支持多种搜索类型名称
            if search_type in ["file", "file_search"]:
                return self._search_files(query, options)
            elif search_type in ["network", "network_search"]:
                return self._search_network(query, options)
            elif search_type in ["system", "system_info"]:
                return self._get_system_info(query, options)
            elif search_type in ["process", "process_search"]:
                return self._search_processes(query, options)
            else:
                # 默认使用文件搜索
                return self._search_files(query, options)
        except Exception as e:
            return f"❌ 搜索执行错误: {str(e)}"
    
    def _search_files(self, query: str, options: Optional[str] = None) -> str:
        """文件搜索"""
        try:
            # 确定搜索路径
            search_path = options if options and options.strip() else "."
            
            # 确保搜索路径是绝对路径
            if not os.path.isabs(search_path):
                search_path = os.path.abspath(search_path)
            
            if not os.path.exists(search_path):
                return f"❌ 搜索路径不存在: {search_path}"
            
            if platform.system() == "Windows":
                # Windows文件搜索 - 支持指定路径
                cmd = f'dir /s /b "{search_path}\\*{query}*"'
            else:
                # Linux/Mac文件搜索
                cmd = f'find {search_path} -name "*{query}*"'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    return f"📁 文件搜索结果 (路径: {search_path}):\n\n{output}"
                else:
                    return f"📁 在路径 {search_path} 中未找到匹配的文件: {query}"
            else:
                # 尝试使用PowerShell进行更精确的搜索
                try:
                    ps_cmd = f'Get-ChildItem -Path "{search_path}" -Recurse -Name "*{query}*"'
                    ps_result = subprocess.run(['powershell', '-Command', ps_cmd], 
                                             capture_output=True, text=True, timeout=60)
                    if ps_result.returncode == 0:
                        output = ps_result.stdout.strip()
                        if output:
                            return f"📁 PowerShell文件搜索结果 (路径: {search_path}):\n\n{output}"
                except:
                    pass
                
                return f"❌ 文件搜索失败: 在路径 {search_path} 中未找到包含 '{query}' 的文件"
        except Exception as e:
            return f"❌ 文件搜索错误: {str(e)}"
    
    def _search_network(self, query: str, options: Optional[str] = None) -> str:
        """网络查询"""
        try:
            if "ping" in query.lower():
                cmd = f"ping {query.replace('ping', '').strip()}"
            elif "nslookup" in query.lower():
                cmd = f"nslookup {query.replace('nslookup', '').strip()}"
            elif "tracert" in query.lower() or "traceroute" in query.lower():
                if platform.system() == "Windows":
                    cmd = f"tracert {query.replace('tracert', '').replace('traceroute', '').strip()}"
                else:
                    cmd = f"traceroute {query.replace('tracert', '').replace('traceroute', '').strip()}"
            else:
                # 默认ping
                cmd = f"ping {query}"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                output = result.stdout.strip()
                return f"🌐 网络查询结果:\n\n{output}"
            else:
                return f"❌ 网络查询失败: {result.stderr}"
        except Exception as e:
            return f"❌ 网络查询错误: {str(e)}"
    
    def _get_system_info(self, query: str, options: Optional[str] = None) -> str:
        """获取系统信息"""
        try:
            if "memory" in query.lower() or "内存" in query:
                if platform.system() == "Windows":
                    cmd = "wmic computersystem get TotalPhysicalMemory"
                else:
                    cmd = "free -h"
            elif "disk" in query.lower() or "磁盘" in query:
                if platform.system() == "Windows":
                    cmd = "wmic logicaldisk get size,freespace,caption"
                else:
                    cmd = "df -h"
            elif "cpu" in query.lower():
                if platform.system() == "Windows":
                    cmd = "wmic cpu get name"
                else:
                    cmd = "lscpu"
            else:
                # 默认系统信息
                if platform.system() == "Windows":
                    cmd = "systeminfo"
                else:
                    cmd = "uname -a"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                output = result.stdout.strip()
                return f"💻 系统信息:\n\n{output}"
            else:
                return f"❌ 获取系统信息失败: {result.stderr}"
        except Exception as e:
            return f"❌ 系统信息查询错误: {str(e)}"
    
    def _search_processes(self, query: str, options: Optional[str] = None) -> str:
        """进程查询"""
        try:
            if platform.system() == "Windows":
                if query:
                    cmd = f'tasklist /fi "imagename eq {query}"'
                else:
                    cmd = "tasklist"
            else:
                if query:
                    cmd = f"ps aux | grep {query}"
                else:
                    cmd = "ps aux"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                output = result.stdout.strip()
                return f"🔄 进程查询结果:\n\n{output}"
            else:
                return f"❌ 进程查询失败: {result.stderr}"
        except Exception as e:
            return f"❌ 进程查询错误: {str(e)}"

class DocumentSearchTool(RecordableTool):
    """文档搜索和读取工具"""
    name: str = "document_search"
    description: str = "搜索本地文档文件（txt、doc、docx）并读取内容"
    
    class InputSchema(BaseModel):
        query: str = Field(description="搜索关键词")
        search_path: str = Field(default=".", description="搜索路径，默认为当前目录")
        max_files: int = Field(default=5, description="最大读取文件数量")
    
    def _original_run(self, query: str = None, search_path: str = ".", max_files: int = 5, **kwargs) -> str:
        """搜索并读取文档"""
        try:
            # 处理参数传递问题 - 修复参数解析
            print(f"DEBUG: 接收到的参数 - query={query}, search_path={search_path}, kwargs={kwargs}")
            
            # 从kwargs中正确提取参数
            if 'keyword' in kwargs:
                query = kwargs['keyword']
            elif 'query' in kwargs:
                query = kwargs['query']
            elif query is None:
                # 如果还是没有，尝试从第一个位置参数获取
                args_list = list(kwargs.values())
                if args_list:
                    query = str(args_list[0])
            
            if 'path' in kwargs:
                search_path = kwargs['path']
            elif 'search_path' in kwargs:
                search_path = kwargs['search_path']
            
            print(f"DEBUG: 解析后的参数 - query={query}, search_path={search_path}")
            
            if not query:
                return "❌ 缺少搜索查询内容"
            
            # 确保搜索路径是绝对路径
            if not os.path.isabs(search_path):
                search_path = os.path.abspath(search_path)
            
            if not os.path.exists(search_path):
                return f"❌ 搜索路径不存在: {search_path}"
            
            # 切换到指定目录进行搜索
            original_cwd = os.getcwd()
            try:
                os.chdir(search_path)
                doc_reader = DocumentReader()
                # 在指定目录下搜索
                matching_files = doc_reader.search_files(".", query)
            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)
            

            
            if not matching_files:
                return f"📁 在路径 {search_path} 中未找到包含关键词 '{query}' 的文档文件"
            
            # 限制文件数量
            matching_files = matching_files[:max_files]
            
            # 第二步：读取文件内容并进行总结
            output = f"📚 文档搜索结果 (路径: {search_path}, 关键词: {query})\n"
            output += f"📊 找到 {len(matching_files)} 个相关文档文件\n\n"
            output += "=" * 60 + "\n\n"
            
            all_content_summary = []
            
            for i, file_info in enumerate(matching_files, 1):
                output += f"📄 文档 {i}: {file_info['name']}\n"
                output += f"   路径: {file_info['path']}\n"
                output += f"   大小: {file_info['size']} 字节\n"
                output += f"   格式: {file_info['extension']}\n"
                
                try:
                    # 读取文件内容
                    content = doc_reader.read_document(file_info['path'])
                    
                    if content.startswith("❌") or content.startswith("无法读取"):
                        output += f"   ❌ 读取错误: {content}\n"
                    else:
                        # 检查内容是否包含搜索关键词
                        if query.lower() in content.lower():
                            output += f"   ✅ 内容包含关键词 '{query}'\n"
                            output += f"   内容长度: {len(content)} 字符\n"
                            
                            # 提取包含关键词的段落
                            paragraphs = content.split('\n\n')
                            relevant_paragraphs = []
                            for para in paragraphs:
                                if query.lower() in para.lower():
                                    relevant_paragraphs.append(para.strip())
                            
                            if relevant_paragraphs:
                                output += f"   相关段落数量: {len(relevant_paragraphs)}\n"
                                output += f"   相关段落预览:\n"
                                for j, para in enumerate(relevant_paragraphs[:3], 1):  # 只显示前3个段落
                                    preview = para[:200] + "..." if len(para) > 200 else para
                                    output += f"      {j}. {preview}\n"
                                
                                # 添加到总总结
                                all_content_summary.append({
                                    'file': file_info['name'],
                                    'path': file_info['path'],
                                    'relevant_content': relevant_paragraphs
                                })
                            else:
                                output += f"   ⚠️ 文件名匹配但内容中未找到关键词\n"
                        else:
                            output += f"   ⚠️ 文件名匹配但内容中未找到关键词\n"
                            
                except Exception as e:
                    output += f"   ❌ 读取错误: {str(e)}\n"
                
                output += "\n" + "-" * 40 + "\n\n"
            
            # 第三步：生成总体总结
            if all_content_summary:
                output += "📋 总体内容总结:\n"
                output += "=" * 40 + "\n\n"
                
                for item in all_content_summary:
                    output += f"📄 文件: {item['file']}\n"
                    output += f"📁 路径: {item['path']}\n"
                    output += f"📝 相关段落数量: {len(item['relevant_content'])}\n"
                    output += f"💡 主要内容:\n"
                    
                    # 为每个文件生成摘要
                    combined_content = "\n\n".join(item['relevant_content'])
                    if len(combined_content) > 500:
                        summary = doc_reader.extract_summary(combined_content, 500)
                        output += f"   {summary}\n"
                    else:
                        output += f"   {combined_content}\n"
                    
                    output += "\n" + "-" * 30 + "\n\n"
                
                output += "💡 提示: 如需查看完整内容，请使用 document_read 工具指定具体文件路径"
            else:
                output += "⚠️ 未找到包含关键词的相关内容"
            
            return output
            
        except Exception as e:
            return f"❌ 文档搜索失败: {str(e)}"

class DocumentReadTool(RecordableTool):
    """文档读取工具"""
    name: str = "document_read"
    description: str = "读取指定路径的文档文件内容"
    
    class InputSchema(BaseModel):
        file_path: str = Field(description="文档文件路径")
        include_summary: bool = Field(default=True, description="是否包含摘要")
    
    def _original_run(self, file_path: str = None, include_summary: bool = True, **kwargs) -> str:
        """读取文档内容"""
        try:
            # 处理参数传递问题
            print(f"DEBUG DocumentReadTool: 接收到的参数 - file_path={file_path}, include_summary={include_summary}, kwargs={kwargs}")
            
            # 从kwargs中正确提取参数
            if 'file_path' in kwargs:
                file_path = kwargs['file_path']
            elif 'path' in kwargs:
                file_path = kwargs['path']
            elif file_path is None:
                # 如果还是没有，尝试从第一个位置参数获取
                args_list = list(kwargs.values())
                if args_list:
                    file_path = str(args_list[0])
            
            print(f"DEBUG DocumentReadTool: 解析后的参数 - file_path={file_path}")
            
            if not file_path:
                return "❌ 缺少文件路径"
            
            doc_reader = DocumentReader()
            
            if not os.path.exists(file_path):
                return f"❌ 文件不存在: {file_path}"
            
            # 检查文件格式
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in doc_reader.supported_extensions:
                return f"❌ 不支持的文件格式: {file_ext}，支持的格式: {', '.join(doc_reader.supported_extensions)}"
            
            # 读取文件内容
            content = doc_reader.read_document(file_path)
            
            if content.startswith("❌") or content.startswith("无法读取"):
                return content
            
            # 格式化输出
            file_info = {
                'name': os.path.basename(file_path),
                'size': os.path.getsize(file_path),
                'extension': file_ext
            }
            
            output = f"📖 文档内容: {file_info['name']}\n"
            output += f"📁 文件路径: {file_path}\n"
            output += f"📊 文件大小: {file_info['size']} 字节\n"
            output += f"📄 文件格式: {file_info['extension']}\n"
            output += f"📝 内容长度: {len(content)} 字符\n"
            output += "=" * 60 + "\n\n"
            
            if include_summary and len(content) > 1000:
                summary = doc_reader.extract_summary(content)
                output += f"📋 内容摘要:\n{summary}\n\n"
                output += "💡 提示: 内容较长，已显示摘要。如需完整内容，请设置 include_summary=False"
            else:
                output += f"📄 完整内容:\n{content}\n"
            
            return output
            
        except Exception as e:
            return f"❌ 文档读取失败: {str(e)}"


# 摄像头工具输入模型
class CameraPhotoInput(BaseModel):
    save_path: Optional[str] = Field(default=None, description="保存路径，留空则使用临时文件")

class CameraRecordInput(BaseModel):
    output_path: str = Field(description="视频保存路径")
    duration: int = Field(default=10, description="录制时长（秒）")

class CameraInfoInput(BaseModel):
    pass

class CameraCloseInput(BaseModel):
    delay_seconds: Optional[int] = Field(default=0, description="延迟关闭时间（秒），0表示立即关闭")

# 摄像头工具类
class CameraPhotoTool(RecordableTool):
    name: str = "camera_photo"
    description: str = "使用摄像头拍照"
    args_schema: Type[BaseModel] = CameraPhotoInput

    def _original_run(self, save_path: str = None, **kwargs) -> str:
        """拍照"""
        try:
            result = camera_manager.take_photo(save_path)
            
            if result['success']:
                return f"✅ 拍照成功\n\n📸 图片路径: {result['file_path']}\n📏 分辨率: {result['width']}x{result['height']}\n⏰ 时间: {result['timestamp']}"
            else:
                return f"❌ 拍照失败: {result['error']}"
                
        except Exception as e:
            return f"❌ 拍照异常: {str(e)}"


class CameraRecordTool(RecordableTool):
    name: str = "camera_record"
    description: str = "使用摄像头录制视频"
    args_schema: Type[BaseModel] = CameraRecordInput

    def _original_run(self, output_path: str, duration: int = 10, **kwargs) -> str:
        """录制视频"""
        try:
            result = camera_manager.start_recording(output_path, duration)
            
            if result['success']:
                return f"✅ 开始录制视频\n\n🎬 输出路径: {result['output_path']}\n⏱️ 时长: {result['duration']}秒\n📏 分辨率: {result['resolution']}\n🎯 帧率: {result['fps']}fps"
            else:
                return f"❌ 录制失败: {result['error']}"
                
        except Exception as e:
            return f"❌ 录制异常: {str(e)}"


class CameraStopRecordTool(RecordableTool):
    name: str = "camera_stop_record"
    description: str = "停止摄像头录制"
    args_schema: Type[BaseModel] = CameraInfoInput

    def _original_run(self, **kwargs) -> str:
        """停止录制"""
        try:
            result = camera_manager.stop_recording()
            
            if result['success']:
                return f"✅ 录制已停止\n\n🎬 视频文件: {result['output_path']}"
            else:
                return f"❌ 停止录制失败: {result['error']}"
                
        except Exception as e:
            return f"❌ 停止录制异常: {str(e)}"


class CameraInfoTool(RecordableTool):
    name: str = "camera_info"
    description: str = "获取摄像头信息"
    args_schema: Type[BaseModel] = CameraInfoInput

    def _original_run(self, **kwargs) -> str:
        """获取摄像头信息"""
        try:
            # 获取可用摄像头列表
            available_cameras = camera_manager.get_available_cameras()
            
            # 获取当前摄像头信息
            info_result = camera_manager.get_camera_info()
            
            result = f"📹 摄像头信息:\n\n"
            result += f"🔍 可用摄像头: {available_cameras}\n\n"
            
            if info_result['success']:
                info = info_result['info']
                result += f"📷 当前摄像头: {info['camera_index']}\n"
                result += f"🔗 连接状态: {'已连接' if info['is_opened'] else '未连接'}\n"
                result += f"📏 分辨率: {info['width']}x{info['height']}\n"
                result += f"🎯 帧率: {info['fps']}fps\n"
                result += f"🎬 录制状态: {'录制中' if info['is_recording'] else '未录制'}\n"
                result += f"💡 亮度: {info['brightness']:.2f}\n"
                result += f"🌓 对比度: {info['contrast']:.2f}\n"
                result += f"🎨 饱和度: {info['saturation']:.2f}"
            else:
                result += f"❌ 获取信息失败: {info_result['error']}"
            
            return result
            
        except Exception as e:
            return f"❌ 获取摄像头信息异常: {str(e)}"


class CameraCloseTool(RecordableTool):
    name: str = "camera_close"
    description: str = "关闭摄像头"
    args_schema: Type[BaseModel] = CameraCloseInput

    def _original_run(self, delay_seconds: int = 0, **kwargs) -> str:
        """关闭摄像头"""
        try:
            if delay_seconds > 0:
                # 延迟关闭
                camera_manager.auto_close_camera(delay_seconds)
                return f"✅ 摄像头将在 {delay_seconds} 秒后自动关闭"
            else:
                # 立即关闭
                close_result = camera_manager.close_camera()
                if close_result['success']:
                    return "✅ 摄像头已关闭"
                else:
                    return f"❌ 关闭摄像头失败: {close_result['error']}"
            
        except Exception as e:
            return f"❌ 关闭摄像头异常: {str(e)}"


# 摄像头拍照并检测工具输入模型
class CameraDetectInput(BaseModel):
    save_path: Optional[str] = Field(default=None, description="图片保存路径，留空则使用临时文件")
    model_id: Optional[str] = Field(default=None, description="指定使用的YOLOv8模型ID")
    confidence: float = Field(default=0.5, description="检测置信度阈值")
    save_results: bool = Field(default=True, description="是否保存检测结果到文件")

# 摄像头拍照并检测工具
class CameraDetectTool(RecordableTool):
    name: str = "camera_detect"
    description: str = "使用摄像头拍照并进行目标检测，支持保存检测结果"
    args_schema: Type[BaseModel] = CameraDetectInput

    def _original_run(self, save_path: str = None, model_id: str = None, confidence: float = 0.5, save_results: bool = True, **kwargs) -> str:
        """拍照并检测"""
        try:
            # 1. 确保摄像头已打开
            if camera_manager.camera is None:
                if not camera_manager.open_camera():
                    return "❌ 无法打开摄像头，请检查摄像头是否可用"
            
            # 2. 拍照
            photo_result = camera_manager.take_photo(save_path)
            if not photo_result['success']:
                return f"❌ 拍照失败: {photo_result['error']}"
            
            image_path = photo_result['file_path']
            
            # 3. 进行目标检测（默认启用绘制边界框和保存标注图片）
            detection_result = local_model_manager.detect_objects(
                image_path, 
                confidence=confidence, 
                model_id=model_id,
                draw_boxes=True,
                show_confidence=True,
                save_annotated=True,
                mask_threshold=0.5
            )
            if not detection_result['success']:
                return f"❌ 检测失败: {detection_result['error']}"
            
            # 4. 生成结果报告
            report = self._generate_detection_report(photo_result, detection_result)
            
            # 5. 保存检测结果（如果需要）
            if save_results:
                results_file = self._save_detection_results(image_path, detection_result, report)
                report += f"\n\n💾 检测结果已保存到: {results_file}"
            
            return report
            
        except Exception as e:
            return f"❌ 拍照检测异常: {str(e)}"
    
    def _generate_detection_report(self, photo_result: dict, detection_result: dict) -> str:
        """生成检测报告"""
        report = f"📸 拍照检测完成\n\n"
        report += f"📷 图片信息:\n"
        report += f"  • 文件路径: {photo_result['file_path']}\n"
        report += f"  • 分辨率: {photo_result['width']}x{photo_result['height']}\n"
        report += f"  • 拍摄时间: {photo_result['timestamp']}\n"
        # 在报告中明确标注图片路径，让前端能够识别
        report += f"  📷 图片路径: {photo_result['file_path']}\n\n"
        
        report += f"🎯 检测结果:\n"
        report += f"  • 使用模型: {detection_result.get('model_used', '未知模型')}\n"
        report += f"  • 置信度阈值: {detection_result.get('confidence_threshold', 0.5)}\n"
        report += f"  • 检测到物体: {detection_result.get('total_objects', 0)} 个\n"
        
        # 如果有标注图片，添加到报告中
        if 'annotated_image' in detection_result:
            report += f"  • 标注图片已保存: {detection_result['annotated_image']}\n"
            # 在报告中明确标注图片路径，让前端能够识别
            report += f"  📷 图片路径: {detection_result['annotated_image']}\n"
        
        report += "\n"
        
        detections = detection_result.get('detections', [])
        if detections:
            report += f"📋 检测详情:\n"
            for i, det in enumerate(detections, 1):
                report += f"  {i}. {det['class']} (置信度: {det['confidence']}%)\n"
                bbox = det['bbox']
                report += f"     位置: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n"
        else:
            report += f"📋 未检测到任何物体\n"
        
        return report
    
    def _save_detection_results(self, image_path: str, detection_result: dict, report: str) -> str:
        """保存检测结果到文件"""
        try:
            import json
            from datetime import datetime
            
            # 创建结果文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"detection_results_{timestamp}.json"
            
            # 准备保存的数据
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "detection_result": detection_result,
                "report": report
            }
            
            # 保存到文件
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            return results_file
            
        except Exception as e:
            return f"保存失败: {str(e)}"


# 视频分析工具输入模型
class VideoAnalysisInput(BaseModel):
    video_path: str = Field(description="视频文件路径")
    frame_interval: int = Field(default=30, description="提取帧的间隔（帧数）")
    model_id: Optional[str] = Field(default=None, description="指定使用的YOLOv8模型ID")
    confidence: float = Field(default=0.5, description="检测置信度阈值")
    save_frames: bool = Field(default=False, description="是否保存提取的帧")

# 视频分析工具
class VideoAnalysisTool(RecordableTool):
    name: str = "video_analysis"
    description: str = "分析视频文件，提取关键帧并进行目标检测"
    args_schema: Type[BaseModel] = VideoAnalysisInput

    def _original_run(self, video_path: str, frame_interval: int = 30, model_id: str = None, confidence: float = 0.5, save_frames: bool = False, **kwargs) -> str:
        """分析视频"""
        try:
            if not os.path.exists(video_path):
                return f"❌ 视频文件不存在: {video_path}"
            
            # 提取视频帧
            frames = self._extract_frames(video_path, frame_interval, save_frames)
            if not frames:
                return "❌ 无法从视频中提取帧"
            
            # 分析每一帧
            analysis_results = []
            for i, frame_path in enumerate(frames):
                detection_result = local_model_manager.detect_objects(
                    frame_path, 
                    confidence=confidence, 
                    model_id=model_id,
                    draw_boxes=True,
                    show_confidence=True,
                    save_annotated=True,
                    mask_threshold=0.5
                )
                if detection_result['success']:
                    analysis_results.append({
                        'frame_index': i * frame_interval,
                        'detections': detection_result['detections'],
                        'total_objects': detection_result['total_objects']
                    })
            
            # 生成分析报告
            report = self._generate_video_analysis_report(video_path, analysis_results, frame_interval)
            
            # 清理临时文件
            if not save_frames:
                for frame_path in frames:
                    try:
                        os.remove(frame_path)
                    except:
                        pass
            
            return report
            
        except Exception as e:
            return f"❌ 视频分析异常: {str(e)}"
    
    def _extract_frames(self, video_path: str, interval: int, save_frames: bool) -> List[str]:
        """提取视频帧"""
        try:
            import cv2
            import tempfile
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    if save_frames:
                        frame_path = f"frame_{extracted_count:04d}.jpg"
                        cv2.imwrite(frame_path, frame)
                    else:
                        # 使用临时文件
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                            frame_path = tmp.name
                            cv2.imwrite(frame_path, frame)
                    
                    frames.append(frame_path)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"提取视频帧失败: {e}")
            return []
    
    def _generate_video_analysis_report(self, video_path: str, analysis_results: List[dict], frame_interval: int) -> str:
        """生成视频分析报告"""
        report = f"🎬 视频分析完成\n\n"
        report += f"📹 视频文件: {video_path}\n"
        report += f"📊 分析帧数: {len(analysis_results)}\n"
        report += f"⏱️ 帧间隔: {frame_interval} 帧\n\n"
        
        if not analysis_results:
            report += "❌ 未检测到任何物体\n"
            return report
        
        # 统计检测结果
        total_detections = sum(result['total_objects'] for result in analysis_results)
        report += f"🎯 总检测数: {total_detections}\n\n"
        
        # 详细结果
        report += "📋 检测详情:\n"
        for result in analysis_results:
            frame_time = result['frame_index'] / 30  # 假设30fps
            report += f"  • 第 {result['frame_index']} 帧 (约 {frame_time:.1f}秒): {result['total_objects']} 个物体\n"
            
            for det in result['detections']:
                report += f"    - {det['class']} (置信度: {det['confidence']}%)\n"
        
        return report


# 本地模型工具输入模型
class ImageAnalysisInput(BaseModel):
    image_path: str = Field(description="图像文件路径")
    analysis_type: str = Field(default="all", description="分析类型: all, classification, detection, faces")
    model_id: Optional[str] = Field(default=None, description="指定使用的YOLOv8模型ID，如yolov8n, yolov8s等")
    confidence: float = Field(default=0.5, description="检测置信度阈值")
    draw_boxes: bool = Field(default=False, description="是否绘制边界框")
    show_confidence: bool = Field(default=True, description="是否显示置信度")
    save_annotated: bool = Field(default=False, description="是否保存标注后的图像")
    mask_threshold: float = Field(default=0.5, description="mask检测阈值")

class ModelListInput(BaseModel):
    pass

class ModelReloadInput(BaseModel):
    pass

class ModelInfoInput(BaseModel):
    pass

# 本地模型工具类
class ImageAnalysisTool(RecordableTool):
    name: str = "image_analysis"
    description: str = "使用本地AI模型分析图像（分类、目标检测、人脸检测）"
    args_schema: Type[BaseModel] = ImageAnalysisInput

    def _original_run(self, image_path: str, analysis_type: str = "all", model_id: str = None, 
                     confidence: float = 0.5, draw_boxes: bool = False, show_confidence: bool = True, 
                     save_annotated: bool = False, mask_threshold: float = 0.5, **kwargs) -> str:
        """图像分析"""
        try:
            if not os.path.exists(image_path):
                return f"❌ 图像文件不存在: {image_path}"
            
            report = f"🖼️ 图像分析完成\n\n"
            report += f"📁 图像路径: {image_path}\n"
            report += f"🔧 分析参数: 置信度={confidence}, 绘制框体={draw_boxes}, 显示置信度={show_confidence}\n\n"
            
            # 根据分析类型执行不同的分析
            if analysis_type in ["all", "detection"]:
                # 目标检测
                detection_result = local_model_manager.detect_objects(
                    image_path, 
                    confidence=confidence, 
                    model_id=model_id,
                    draw_boxes=draw_boxes,
                    show_confidence=show_confidence,
                    save_annotated=save_annotated,
                    mask_threshold=mask_threshold
                )
                
                if detection_result['success']:
                    detections = detection_result.get('detections', [])
                    report += f"🎯 目标检测结果:\n"
                    report += f"  • 使用模型: {detection_result.get('model_used', '未知')}\n"
                    report += f"  • 检测到物体: {len(detections)} 个\n"
                    
                    if detections:
                        for i, det in enumerate(detections, 1):
                            report += f"  {i}. {det['class']} (置信度: {det['confidence']}%)\n"
                            if 'mask_area' in det:
                                report += f"     Mask面积: {det['mask_area']} 像素\n"
                    
                    if 'annotated_image' in detection_result:
                        report += f"  • 标注图像已保存: {detection_result['annotated_image']}\n"
                        # 在报告中明确标注图片路径，让前端能够识别
                        report += f"  📷 图片路径: {detection_result['annotated_image']}\n"
                    
                    report += "\n"
                else:
                    report += f"❌ 目标检测失败: {detection_result['error']}\n\n"
            
            if analysis_type in ["all", "classification"]:
                # 图像分类
                classification_result = local_model_manager.classify_image(image_path)
                if classification_result['success']:
                    classifications = classification_result.get('classifications', [])
                    report += f"🏷️ 图像分类结果:\n"
                    for i, cls in enumerate(classifications[:3], 1):  # 显示前3个分类
                        report += f"  {i}. {cls['label']} (置信度: {cls['confidence']}%)\n"
                    report += "\n"
                else:
                    report += f"❌ 图像分类失败: {classification_result['error']}\n\n"
            
            if analysis_type in ["all", "faces"]:
                # 人脸检测
                face_result = local_model_manager.detect_faces(image_path)
                if face_result['success']:
                    face_count = face_result.get('face_count', 0)
                    report += f"👤 人脸检测结果: 检测到 {face_count} 张人脸\n\n"
                else:
                    report += f"❌ 人脸检测失败: {face_result['error']}\n\n"
            
            return report
            
        except Exception as e:
            return f"❌ 图像分析异常: {str(e)}"


class ModelInfoTool(RecordableTool):
    name: str = "model_info"
    description: str = "获取本地AI模型信息"
    args_schema: Type[BaseModel] = ModelInfoInput

    def _original_run(self, **kwargs) -> str:
        """获取模型信息"""
        try:
            result = local_model_manager.get_model_info()
            
            if result['success']:
                info = result['info']
                
                output = f"🤖 本地AI模型信息:\n\n"
                output += f"💻 设备: {info['device']}\n"
                output += f"🚀 CUDA支持: {'是' if info['cuda_available'] else '否'}\n"
                output += f"📦 PyTorch版本: {info['torch_version']}\n"
                output += f"🖼️ TorchVision版本: {info['torchvision_version']}\n"
                output += f"🎯 默认模型: {info['default_model']}\n\n"
                
                # 已加载模型信息
                loaded_models = info.get('loaded_models', [])
                if loaded_models:
                    output += f"📋 已加载模型 ({len(loaded_models)}个):\n"
                    for model in loaded_models:
                        output += f"  • {model['name']} ({model['type']})\n"
                
                # 本地模型摘要
                local_summary = info.get('local_models_summary', {})
                if local_summary:
                    output += f"\n📁 本地模型配置:\n"
                    output += f"  总配置数: {local_summary.get('total_models', 0)}\n"
                    output += f"  可用模型: {local_summary.get('available_count', 0)}\n"
                    output += f"  缺失模型: {local_summary.get('missing_count', 0)}\n"
                
                if info['cuda_available']:
                    output += f"\n🎮 GPU: {info['gpu_name']}\n"
                    output += f"💾 GPU内存: {info['gpu_memory'] / 1024**3:.1f}GB"
                
                return output
            else:
                return f"❌ 获取模型信息失败: {result['error']}"
                
        except Exception as e:
            return f"❌ 获取模型信息异常: {str(e)}"


class ModelListTool(RecordableTool):
    name: str = "model_list"
    description: str = "获取本地可用模型列表"
    args_schema: Type[BaseModel] = ModelListInput

    def _original_run(self, **kwargs) -> str:
        """获取模型列表"""
        try:
            result = local_model_manager.get_available_model_list()
            
            if result['success']:
                models = result['models']
                
                output = f"📋 本地可用模型列表 ({result['total_count']}个):\n\n"
                
                for model in models:
                    status_icon = "✅" if model['status'] == 'available' else "❌"
                    file_size = model.get('file_size', 0)
                    size_str = f"{file_size / 1024**2:.1f}MB" if file_size > 0 else "未知"
                    
                    output += f"{status_icon} {model['name']} ({model['id']})\n"
                    output += f"   类型: {model['type']} | 任务: {model['task']}\n"
                    output += f"   描述: {model['description']}\n"
                    output += f"   状态: {model['status']} | 大小: {size_str}\n"
                    output += f"   路径: {model['file_path']}\n\n"
                
                return output
            else:
                return f"❌ 获取模型列表失败: {result['error']}"
                
        except Exception as e:
            return f"❌ 获取模型列表异常: {str(e)}"


class ModelReloadTool(RecordableTool):
    name: str = "model_reload"
    description: str = "重新加载本地模型"
    args_schema: Type[BaseModel] = ModelReloadInput

    def _original_run(self, **kwargs) -> str:
        """重新加载模型"""
        try:
            result = local_model_manager.reload_models()
            
            if result['success']:
                return f"✅ 模型重新加载成功\n\n🔄 已加载 {result['loaded_count']} 个模型"
            else:
                return f"❌ 模型重新加载失败: {result['error']}"
                
        except Exception as e:
            return f"❌ 模型重新加载异常: {str(e)}"


class UnifiedLangChainAgent:
    """统一的LangChain AI Agent"""
    
    def __init__(self):
        """初始化统一的LangChain Agent"""
        try:
            self.llm = ChatOpenAI(
                model=Config.OPENAI_MODEL,
                openai_api_key=Config.OPENAI_API_KEY,
                openai_api_base=Config.OPENAI_BASE_URL,
                temperature=0.7
            )
            
            # 定义所有工具
            self.tools = [
                AudioRecordTool(),
                AudioPlayTool(),
                TextToSpeechTool(),
                WebSearchTool(),
                WebSummaryTool(),
                BrowserTool(),
                SystemCommandTool(),
                FileOperationTool(),
                CommandLineSearchTool(),
                DocumentSearchTool(),
                DocumentReadTool(),
                CameraPhotoTool(),
                CameraRecordTool(),
                CameraStopRecordTool(),
                CameraInfoTool(),
                CameraCloseTool(),
                CameraDetectTool(),
                VideoAnalysisTool(),
                ImageAnalysisTool(),
                ModelInfoTool(),
                ModelListTool(),
                ModelReloadTool()
            ]
            
            # 初始化Agent
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )
            
            # 系统提示词
            self.system_prompt = """【重要说明】所有总结、输出、回答都必须用中文！

你是一个智能AI助手，具有以下能力：

1. **音频处理**：录制音频、播放音频、文本转语音
2. **网络功能**：网页搜索、搜索并总结、打开浏览器
3. **本地工具**：执行系统命令、文件操作、命令行搜索、文档搜索、文档读取
4. **摄像头功能**：拍照、录制视频、获取摄像头信息
5. **AI模型分析**：图像分类、目标检测、人脸检测
6. **本地模型管理**：模型列表、模型信息、模型重载

**智能决策**：
- 搜索相关：使用 web_search 或 web_summary
- 文档搜索：使用 document_search（可指定搜索路径）
- 文件搜索：使用 cmd_search（可指定搜索路径）
- 音频相关：使用 audio_record、audio_play、text_to_speech
- 系统命令：使用 system_command
- 摄像头相关：使用 camera_photo、camera_record、camera_stop_record、camera_info、camera_close（关闭摄像头）、camera_detect（拍照并检测）、video_analysis（视频分析）
- 图像分析：使用 image_analysis（支持分类、检测、人脸识别，可指定模型ID）
- 模型管理：使用 model_info、model_list、model_reload 管理本地AI模型

**图像检测重要说明**：
- 当用户要求进行目标检测或图像分析时，**必须设置 draw_boxes=True 和 save_annotated=True**
- 这样可以生成带有边界框标注的图片，让用户能够直观看到检测结果
- 检测结果会包含边界框位置信息，格式为 [x1, y1, x2, y2]
- 标注图片会自动保存并显示在前端界面中

**重要说明**：
- 所有回答都必须用中文
- 搜索工具支持指定搜索路径，不限于D盘
- 在回答中标注信息来源和链接
- 提供准确、客观的信息总结
- 摄像头功能需要用户确认权限
- 图像分析功能需要本地AI模型支持

请根据用户需求选择合适的工具完成任务。"""
            
        except Exception as e:
            print(f"❌ 统一LangChain Agent初始化失败: {e}")
            self.agent = None

    def chat(self, message: str) -> str:
        """与AI Agent对话"""
        try:
            if not self.agent:
                return "❌ 统一LangChain Agent未正确初始化，请检查配置"
            
            # 构建完整的提示词
            full_prompt = f"{self.system_prompt}\n\n用户消息: {message}\n\n请根据用户需求，选择合适的工具来完成任务。"
            
            # 执行Agent
            response = self.agent.run(full_prompt)
            return response
            
        except Exception as e:
            return f"❌ Agent执行错误: {str(e)}"
    
    def chat_with_tool_calls(self, message: str) -> Dict:
        """与AI Agent对话，并记录工具调用信息"""
        try:
            if not self.agent:
                return {
                    'success': False,
                    'response': "❌ 统一LangChain Agent未正确初始化，请检查配置",
                    'tool_calls': []
                }
            
            # 清空之前的工具调用记录
            clear_tool_calls()
            
            # 构建完整的提示词
            full_prompt = f"{self.system_prompt}\n\n用户消息: {message}\n\n请根据用户需求，选择合适的工具来完成任务。"
            
            # 执行Agent
            response = self.agent.run(full_prompt)
            
            # 获取工具调用记录
            tool_calls = get_tool_calls()
            
            return {
                'success': True,
                'response': response,
                'tool_calls': tool_calls
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': f"❌ Agent执行错误: {str(e)}",
                'tool_calls': get_tool_calls()
            }
    
    def get_tool_call_info(self, message: str) -> dict:
        """获取工具调用信息（用于前端显示）"""
        try:
            if not self.agent:
                return {"error": "Agent未初始化"}
            
            # 这里可以返回工具调用的详细信息
            # 由于LangChain的Agent执行是内部的，我们返回基本信息
            return {
                "available_tools": [tool.name for tool in self.tools],
                "message": message,
                "status": "ready"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_available_tools(self) -> List[Dict]:
        """获取可用工具列表"""
        if not self.tools:
            return []
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "type": "unified_tool"
            }
            for tool in self.tools
        ]

# 全局实例
unified_agent = UnifiedLangChainAgent() 
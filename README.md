# AI Agent 应用

基于OpenAI API的智能AI Agent应用，支持语音交互、网络搜索、本地工具调用等功能。

## 🚀 功能特性

- **语音交互**: 支持麦克风录音和文本转语音播放
- **网络搜索**: 多源搜索（DuckDuckGo、Bing、Baidu）并返回详细结果
- **本地工具**: 文件搜索、系统命令执行、网络查询等
- **Web界面**: 现代化的Flask Web界面，支持实时对话
- **模型选择**: 支持多种OpenAI模型切换
- **对话历史**: 完整的对话历史记录和记忆功能
- **一键启动**: 支持打包为exe文件，双击即可运行

## 📋 系统要求

- Windows 11/10
- Python 3.8+
- 麦克风和扬声器
- OpenAI API密钥

## 🛠️ 安装和配置

### 1. 克隆项目
```bash
git clone <repository-url>
cd OPENAI-Agent
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
创建 `.env` 文件并设置以下变量：
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo
OPENAI_BASE_URL=https://aiapi.dmid.cc/v1
```

## 🎯 使用方法

### 方法一：直接运行Python脚本
```bash
python main.py
```
应用将自动启动Web服务并打开浏览器。

### 方法二：构建exe文件（推荐）
```bash
# 使用批处理文件（Windows）
build.bat

# 或使用Python脚本
python build_simple.py
```

构建完成后，双击 `dist/AI_Agent.exe` 即可运行。

## 🔧 主要功能

### 1. 语音交互
- **录音**: 使用 `audio_record` 工具录制音频
- **播放**: 使用 `audio_play` 工具播放音频文件
- **TTS**: 使用 `text_to_speech` 工具将文本转换为语音

### 2. 网络搜索
- **多源搜索**: 同时搜索DuckDuckGo、Bing、Baidu
- **详细结果**: 返回标题、摘要、链接和内容
- **智能总结**: AI自动总结搜索结果

### 3. 本地工具
- **文件搜索**: 搜索D盘所有文件和文件夹
- **系统命令**: 执行Windows命令（带安全检查）
- **网络查询**: 查询网络连接和系统信息
- **进程管理**: 查询和管理系统进程
- **文档读取**: 搜索和读取本地文档文件（txt、doc、docx）

### 4. 安全特性
- **命令确认**: 危险命令需要用户确认
- **安全检查**: 禁止执行格式化、删除等危险操作
- **关机保护**: 关机命令需要二次确认

## 📁 项目结构

```
OPENAI-Agent/
├── main.py                 # 主启动脚本
├── app_flask.py           # Flask Web应用
├── langchain_agent.py     # 统一AI Agent
├── config.py              # 配置文件
├── requirements.txt       # Python依赖
├── build_simple.py        # exe构建脚本
├── build.bat              # Windows构建批处理
├── templates/             # Web模板
│   └── index.html         # 主页面
├── tools/                 # 工具模块
│   └── web_search.py      # 网络搜索工具
└── dist/                  # 构建输出目录
    └── AI_Agent.exe       # 可执行文件
```

## 🎮 使用示例

### 基本对话
```
用户: 你好，请介绍一下自己
AI: 你好！我是一个基于OpenAI的AI助手，可以帮助你进行语音交互、网络搜索、文件操作等任务...
```

### 网络搜索
```
用户: 搜索最新的AI技术发展
AI: 🌐 搜索 '最新的AI技术发展' 的结果:
📊 查询关键词: 最新的AI技术发展
📋 找到 5 个相关结果
📚 参考资料源:
📖 资料源 1:
   标题: 2024年AI技术发展趋势
   来源: DuckDuckGo
   链接: https://example.com/ai-trends-2024
   摘要: 2024年AI技术的主要发展方向...
```

### 文件搜索
```
用户: 搜索D盘中的Python文件
AI: 🔍 在D盘中搜索Python文件...
✅ 找到以下Python文件:
D:\projects\test.py
D:\scripts\main.py
...
```

### 文档读取
```
用户: 搜索包含"项目计划"的文档
AI: 📚 文档搜索结果: 找到 2 个相关文档
📄 文档 1: 项目计划.txt
   路径: D:\projects\项目计划.txt
   大小: 2048 字节
   格式: .txt
   内容长度: 1500 字符
   摘要预览: 项目概述：这是一个关于AI助手开发的项目计划...

用户: 读取 D:\projects\项目计划.txt 的内容
AI: 📖 文档内容: 项目计划.txt
📁 文件路径: D:\projects\项目计划.txt
📊 文件大小: 2048 字节
📄 文件格式: .txt
📝 内容长度: 1500 字符
📋 内容摘要:
项目概述：这是一个关于AI助手开发的项目计划...
```

### 系统命令
```
用户: 执行关机命令
AI: ⚠️ 即将执行关机命令: shutdown /s /t 60
请确认您真的要关闭计算机吗？
如果确认，请再次发送相同的命令。
```

## ⚙️ 配置选项

### 模型选择
在Web界面中可以切换不同的OpenAI模型：
- gpt-4-turbo
- gpt-4
- gpt-3.5-turbo
- 其他自定义模型

### API配置
支持自定义OpenAI API地址，适用于不同的API提供商。

### 语音设置
- 自动播放AI回答语音（可开关）
- 手动语音播放按钮
- 可调节语音速度和音量

## 🔒 安全注意事项

1. **API密钥**: 请妥善保管您的OpenAI API密钥
2. **系统命令**: 应用会阻止危险的系统命令执行
3. **文件访问**: 文件搜索仅限于D盘，不会访问系统关键目录
4. **网络访问**: 网络搜索使用公开的搜索引擎API

## 🐛 故障排除

### 常见问题

1. **应用无法启动**
   - 检查Python版本（需要3.8+）
   - 确认所有依赖已安装
   - 检查.env文件配置

2. **语音功能异常**
   - 确认麦克风和扬声器正常工作
   - 检查音频驱动是否正常
   - 尝试重新安装pyaudio和pyttsx3

3. **网络搜索失败**
   - 检查网络连接
   - 确认防火墙设置
   - 尝试使用不同的搜索引擎

4. **exe文件无法运行**
   - 确认Windows Defender未阻止
   - 检查是否有杀毒软件误报
   - 尝试以管理员身份运行

### 日志查看
应用运行时会显示详细的日志信息，包括：
- 服务启动状态
- API调用结果
- 工具执行情况
- 错误信息

## 📄 许可证

本项目仅供学习和研究使用，请遵守相关法律法规和OpenAI的使用条款。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 📞 支持

如果您遇到问题或有建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 使用本应用前请确保您已阅读并同意OpenAI的使用条款。

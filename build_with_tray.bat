@echo off
echo 🔨 开始构建带系统托盘的AI Agent exe文件...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)

REM 安装依赖
echo 📦 安装依赖包...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)

REM 构建exe
echo 🔨 构建exe文件...
python build_simple.py
if errorlevel 1 (
    echo ❌ 构建失败
    pause
    exit /b 1
)

echo.
echo 🎉 构建完成！
echo 📁 exe文件位置: dist\AI_Agent.exe
echo.
echo 📋 使用说明:
echo 1. 双击运行 dist\AI_Agent.exe
echo 2. 应用将在后台运行，系统托盘会显示AI图标
echo 3. 右键点击托盘图标可以打开浏览器、查看日志或退出
echo 4. 确保.env文件中设置了正确的OpenAI API密钥
echo.
pause 
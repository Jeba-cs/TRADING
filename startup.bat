@echo off
echo 🚀 Starting Smart AI Trading System...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create .env if it doesn't exist
if not exist ".env" (
    echo ⚙️ Creating .env file...
    copy .env.example .env
    echo ✏️  Please edit .env file with your API keys
)

echo ✅ Setup complete!
echo 🎯 To start the app, run: streamlit run main.py
pause

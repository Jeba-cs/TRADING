#!/bin/bash
# startup.sh - Quick startup script

echo "🚀 Starting Smart AI Trading System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your API keys"
fi

echo "✅ Setup complete!"
echo "🎯 To start the app, run: streamlit run main.py"

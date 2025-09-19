# 🚀 QUICK START GUIDE

## ⚡ 1-Minute Setup

### Option A: Automatic Setup (Recommended)
```bash
# Linux/Mac
chmod +x startup.sh
./startup.sh

# Windows  
startup.bat
```

### Option B: Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure (optional)
cp .env.example .env
# Edit .env with your API keys

# Run the app
streamlit run main.py
```

## 🎯 First Steps

1. **Open your browser** to http://localhost:8501
2. **Try the AI Assistant** - Ask: "Should I buy AAPL?"
3. **Check Agent Consensus** on the Dashboard tab
4. **Explore Analytics** for market insights
5. **Configure Settings** for your preferences

## 💬 Example Questions for AI

- "Should I buy TESLA stock?"
- "What's the market outlook today?"  
- "Analyze APPLE for swing trading"
- "Is my portfolio too risky?"
- "Explain current market trends"

## 🛡️ Safety First

- Start with **Paper Trading** mode
- Never invest more than you can afford to lose
- This is **educational software** - not financial advice
- Do your own research before making decisions

## 🆘 Having Issues?

1. **Python version**: Ensure Python 3.8+ is installed
2. **Dependencies**: Run `pip install -r requirements.txt` again
3. **Port conflict**: Use `streamlit run main.py --server.port 8502`
4. **API errors**: Check your .env file configuration

**🎉 Happy Trading!** 📈

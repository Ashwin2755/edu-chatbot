# 🚀 Mega AI - Educational Chatbot

An advanced AI-powered educational chatbot built with FastAPI, Google Gemini, and OpenRouter.

## Features

- 🤖 **Multi-Model AI** - Intelligent routing between Gemini Flash, Pro, Mistral, and LLaMA
- 💬 **Conversation History** - Persistent chat sessions
- 📄 **Document Upload** - PDF, Word, Excel support with RAG
- 🔐 **User Authentication** - JWT-based login/signup
- ⚡ **Response Caching** - Optimized performance
- 🎨 **Modern UI** - Clean, responsive interface

## Local Development

### 1. Clone & Setup
```bash
git clone https://github.com/Ashwin2755/edu-chatbot.git
cd edu-chatbot
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure Environment
Create `backend/.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
SITE_URL=http://localhost:8000
SITE_NAME=Mega AI
DEBUG=False
LOG_LEVEL=INFO
```

### 4. Run
```bash
uvicorn openrouter_main:app --reload --port 8000
```

Visit: http://localhost:8000

---

## 🌐 Deploy to Render (Free)

### Option 1: One-Click Deploy
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Option 2: Manual Deploy

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Go to [Render.com](https://render.com)** → Sign up/Login

3. **Create New Web Service**
   - Connect your GitHub repo: `Ashwin2755/edu-chatbot`
   - Settings:
     - **Root Directory**: `backend`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn openrouter_main:app --host 0.0.0.0 --port $PORT`

4. **Add Environment Variables** (in Render dashboard):
   | Key | Value |
   |-----|-------|
   | `GEMINI_API_KEY` | Your Gemini API key |
   | `OPENROUTER_API_KEY` | Your OpenRouter API key |
   | `SITE_URL` | `https://your-app.onrender.com` |
   | `SITE_NAME` | `Mega AI` |

5. **Deploy!** 🎉

---

## 🚂 Deploy to Railway (Alternative)

1. Go to [Railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select `Ashwin2755/edu-chatbot`
4. Add environment variables
5. Deploy!

---

## API Keys

| Service | Get Key From |
|---------|--------------|
| Google Gemini | https://aistudio.google.com/app/apikey |
| OpenRouter | https://openrouter.ai/keys |

---

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **AI**: Google Gemini, OpenRouter (Mistral, LLaMA, DeepSeek)
- **Database**: SQLite
- **Auth**: JWT + bcrypt
- **RAG**: ChromaDB, Sentence Transformers

## License

MIT License

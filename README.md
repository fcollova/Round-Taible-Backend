# Round TAIble Python Backend

This is the Python backend for Round TAIble that interfaces with real LLM models through OpenRouter.

## Setup

1. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Configure OpenRouter API**:
   - Edit `../config.conf` 
   - Replace `your_openrouter_api_key_here` with your actual OpenRouter API key
   - Get your API key from: https://openrouter.ai/

3. **Start the server**:
   ```bash
   python main.py
   ```

The server will start on `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status with available models
- `GET /models` - List available LLM models
- `POST /chat/completions` - General chat completions endpoint
- `POST /debate/generate` - Generate debate messages with context

## Configuration

All configuration is managed in `../config.conf`:

- OpenRouter API credentials
- Model mappings (GPT-4, Claude, Gemini, Llama)
- Server settings
- CORS configuration

## Models

The backend supports 4 real LLM models through OpenRouter:
- **GPT-4** (OpenAI)
- **Claude** (Anthropic) 
- **Gemini** (Google)
- **Llama 3** (Meta)

Each model receives context-aware prompts based on the debate topic and previous messages.
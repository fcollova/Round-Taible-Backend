# 🚀 Deploy Round TAIble Backend su Render

Questa cartella contiene tutto il necessario per deployare il backend su Render.com gratuitamente.

## **📋 Quick Start (5 minuti)**

### **1. Prepara Repository**
```bash
# Dalla cartella deploy-render
cd /home/francesco/Roud-TAIble/deploy-render

# Crea repository GitHub separato per il backend
git init
git add .
git commit -m "Round TAIble Backend for Render"

# Crea repo su GitHub e push
git remote add origin https://github.com/your-username/round-taible-backend.git
git push -u origin main
```

### **2. Deploy su Render**
1. Vai su **https://render.com**
2. Clicca **"New +"** → **"Web Service"**
3. Connetti il repository GitHub `round-taible-backend`
4. Configurazione automatica:

```
Name: round-taible-backend
Branch: main
Root Directory: (lascia vuoto)
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### **3. Configura Environment Variables**
Nel dashboard Render → Environment:

**Required:**
```
OPENROUTER_API_KEY = your-openrouter-key-here
```

**Optional (hanno defaults):**
```
PYTHON_VERSION = 3.11
OPENROUTER_BASE_URL = https://openrouter.ai/api/v1
OPENROUTER_TIMEOUT = 60
```

### **4. Deploy!**
- Clicca **"Create Web Service"**
- Render farà automaticamente build e deploy
- URL backend: `https://round-taible-backend.onrender.com`

---

## **✅ Cosa è incluso in questa cartella**

### **File modificati per Render:**
- ✅ `main.py` - Configurazione compatibile Render
- ✅ `requirements.txt` - Dependencies Python
- ✅ `render.yaml` - Configurazione Render (opzionale)
- ✅ `.env.example` - Template environment variables
- ✅ Tutti i file backend originali

### **Modifiche applicate:**
- ✅ **Fallback configuration**: Funziona senza config.conf
- ✅ **Environment variables**: Legge da OS env vars
- ✅ **CORS aggiornato**: Include *.onrender.com
- ✅ **Model defaults**: Fallback se config.conf manca
- ✅ **Error handling**: Gestisce missing files

---

## **🔧 Configurazione Dettagliata**

### **Environment Variables su Render:**

| Variable | Value | Description |
|----------|--------|-------------|
| `OPENROUTER_API_KEY` | `sk-or-v1-xxx` | **REQUIRED** - Tua API key OpenRouter |
| `PYTHON_VERSION` | `3.11` | Versione Python (default) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Base URL API (default) |
| `OPENROUTER_TIMEOUT` | `60` | Timeout requests in secondi (default) |

### **Free Tier Limits:**
- ✅ **750 ore/mese** di runtime
- ✅ **512MB RAM**
- ✅ **Auto-sleep** dopo 15min inattività
- ✅ **Auto-wake** su richieste
- ✅ **SSL gratuito**
- ✅ **Custom domains**

---

## **🌐 Integrazione con Frontend**

### **URL Backend:**
```
https://round-taible-backend.onrender.com
```

### **Configura Frontend (Vercel):**
```bash
# Environment Variables Vercel
NEXT_PUBLIC_BACKEND_URL=https://round-taible-backend.onrender.com
```

### **Test Endpoints:**
```bash
# Health check
curl https://round-taible-backend.onrender.com/models

# WebSocket test (dal browser)
new WebSocket('wss://round-taible-backend.onrender.com/ws/debates/test')
```

---

## **📊 Monitoring**

### **Dashboard Render:**
- **Logs**: Real-time logs dell'applicazione
- **Metrics**: CPU, Memory, Request count
- **Events**: Deploy history, errors
- **Settings**: Environment vars, custom domains

### **Logs Debugging:**
```bash
# Common issues to check in Render logs:

✅ "Application startup complete" - OK
❌ "ModuleNotFoundError" - Missing dependency
❌ "Configuration error" - Missing environment var
❌ "Port binding error" - Wrong start command
```

---

## **🚨 Troubleshooting**

### **Deploy Fails:**
```bash
# Check build logs in Render dashboard
# Common fixes:

1. requirements.txt missing dependency
2. Python version mismatch
3. Start command incorrect
```

### **App Crashes:**
```bash
# Check application logs
# Common issues:

1. OPENROUTER_API_KEY missing
2. CORS configuration
3. WebSocket connection errors
```

### **Slow Response:**
```bash
# Free tier limitations:
- App sleeps after 15min inactivity
- First request dopo sleep: ~30 secondi
- Soluzioni:
  1. Upgrade a paid plan ($7/mese)
  2. Keep-alive ping ogni 10 minuti
  3. UptimeRobot per auto-ping
```

---

## **⬆️ Upgrade Options**

### **Starter Plan ($7/mese):**
- ✅ **No sleep** - Always-on
- ✅ **Faster builds**
- ✅ **More resources**
- ✅ **Priority support**

### **Keep Free Tier Active:**
```bash
# Setup UptimeRobot (gratis)
# Ping ogni 5 minuti: https://your-backend.onrender.com/models
# Mantiene app sveglia durante il giorno
```

---

## **🔄 Auto-Deploy**

### **Setup completato:**
- ✅ **Auto-deploy** da GitHub main branch
- ✅ **Build automatico** ad ogni push
- ✅ **Rollback** disponibile in dashboard
- ✅ **Preview deployments** per PR

### **Workflow:**
```bash
# 1. Modifica codice localmente
git add .
git commit -m "Update backend"
git push

# 2. Render rileva push automaticamente
# 3. Build e deploy automatico
# 4. App aggiornata in ~2-3 minuti
```

---

## **🎯 Next Steps**

1. ✅ **Deploy backend** su Render
2. ✅ **Test endpoints** con curl
3. ✅ **Configura frontend** con nuovo backend URL
4. ✅ **Test completo** workflow dibattito
5. ✅ **Setup monitoring** (opzionale)

**🚀 Il tuo backend è pronto per produzione su Render!**

---

## **💡 Tips per Performance**

### **Ottimizzazione Free Tier:**
```python
# Nel codice, aggiungi keep-alive endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Setup UptimeRobot per ping /health ogni 5 minuti
```

### **Monitoring Performance:**
```bash
# Setup Sentry (gratis) per error tracking
pip install sentry-sdk[fastapi]

# main.py
import sentry_sdk
sentry_sdk.init(dsn="your-sentry-dsn")
```

### **Database Connection:**
```python
# Se usi database, ottimizza connection pooling
# Render ha PostgreSQL gratuito (512MB)
```

**Il backend è completamente pronto per Render! 🎉**
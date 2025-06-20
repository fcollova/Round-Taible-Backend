# 🌐 Configurazione Frontend URLs

Il backend ora supporta configurazione dinamica degli URL del frontend per API calls e CORS.

## 📋 Configurazioni Disponibili

### 1. File `config.conf`
```ini
[frontend]
# Frontend API URL per operazioni database
url = http://localhost:3000
# Timeout per chiamate API al frontend (secondi)
timeout = 10
```

### 2. Variabili Ambiente (priorità superiore)
```bash
export FRONTEND_URL=https://yourdomain.com
export FRONTEND_TIMEOUT=15
```

## 🔧 Utilizzo

### Sviluppo Locale
```bash
# Avvia frontend
npm run dev  # http://localhost:3000

# Avvia backend (usa config.conf automaticamente)
python main.py  # Userà http://localhost:3000
```

### Produzione
```bash
# Imposta URL di produzione
export FRONTEND_URL=https://round-taible.vercel.app
export FRONTEND_TIMEOUT=20

# Avvia backend
python main.py
```

## 📊 Dove Viene Utilizzato

1. **Salvataggio Messaggi**: `POST {FRONTEND_URL}/api/debates/{id}/messages`
2. **Caricamento Modelli**: `GET {FRONTEND_URL}/api/models`
3. **CORS Origins**: Permette richieste da `{FRONTEND_URL}`
4. **HTTP Referer**: Header per OpenRouter API

## 🧪 Test Configurazione

```bash
# Test configurazione attuale
python test_config.py

# Test con variabili custom
FRONTEND_URL=https://test.com python test_config.py
```

## 🚀 Deploy Produzione

### Render.com
Il file `render.yaml` è già configurato con:
```yaml
envVars:
  - key: FRONTEND_URL
    value: "https://round-taible.vercel.app"
  - key: FRONTEND_TIMEOUT
    value: "15"
```

### Altri Servizi
Imposta le variabili ambiente:
- `FRONTEND_URL`: URL completo del frontend (es: https://yourdomain.com)
- `FRONTEND_TIMEOUT`: Timeout in secondi per API calls

## 🔄 Backward Compatibility

Se nessuna configurazione è presente, il backend usa automaticamente:
- URL: `http://localhost:3000` 
- Timeout: `10` secondi

## 📝 Esempi di Configurazione

### Vercel + Render
```bash
# Frontend: https://round-taible.vercel.app
# Backend: https://round-taible-backend.onrender.com
export FRONTEND_URL=https://round-taible.vercel.app
```

### Railway + Railway
```bash
# Frontend: https://round-taible-production.up.railway.app
# Backend: https://round-taible-backend-production.up.railway.app
export FRONTEND_URL=https://round-taible-production.up.railway.app
```

### Custom Domain
```bash
# Frontend: https://app.roundtaible.com
# Backend: https://api.roundtaible.com
export FRONTEND_URL=https://app.roundtaible.com
```

## ⚙️ File Modificati

- `config.conf`: Aggiunta sezione `[frontend]`
- `main.py`: Sostituiti hardcoded localhost:3000
- `llm_queue_manager.py`: Reso configurabile frontend URL
- `render.yaml`: Aggiunte variabili ambiente produzione

## 🎯 Benefici

✅ **Flessibilità**: URL configurabile per ogni ambiente  
✅ **Deploy Facile**: Nessun hardcoding da cambiare  
✅ **Sicurezza**: CORS configurabile dinamicamente  
✅ **Testing**: Test con domini diversi  
✅ **Produzione**: Pronto per deploy su qualsiasi servizio
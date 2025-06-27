#!/usr/bin/env python3
"""
Script per testare le credenziali OpenRouter
"""

import requests
import json
from config_manager import get_config

def test_openrouter_credentials():
    """Test delle credenziali OpenRouter"""
    print("🔑 Test Credenziali OpenRouter")
    print("=" * 50)
    
    # Carica configurazione
    try:
        config = get_config()
        api_key = config.get_openrouter_api_key()
        base_url = config.get_openrouter_base_url()
        
        print(f"Environment: {config.get_environment()}")
        print(f"Base URL: {base_url}")
        print(f"API Key: {api_key[:20]}..." if api_key else "❌ API Key vuota!")
        print()
        
        if not api_key:
            print("❌ ERRORE: API Key non trovata nella configurazione!")
            return False
            
    except Exception as e:
        print(f"❌ ERRORE nel caricamento configurazione: {e}")
        return False
    
    # Test 1: Verifica modelli disponibili
    print("📋 Test 1: Lista modelli disponibili")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{base_url}/models",
            headers=headers,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Trovati {len(models.get('data', []))} modelli")
            
            # Cerca il modello che sta fallendo
            target_model = "meta-llama/llama-3.1-8b-instruct:free"
            found_model = None
            for model in models.get('data', []):
                if model.get('id') == target_model:
                    found_model = model
                    break
            
            if found_model:
                print(f"✅ Modello {target_model} trovato")
                print(f"   Context Length: {found_model.get('context_length', 'N/A')}")
                print(f"   Pricing: {found_model.get('pricing', 'N/A')}")
            else:
                print(f"❌ Modello {target_model} NON trovato")
                print("📋 Primi 5 modelli disponibili:")
                for i, model in enumerate(models.get('data', [])[:5]):
                    print(f"   {i+1}. {model.get('id', 'N/A')}")
                    
        else:
            print(f"❌ Errore: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ ERRORE nella richiesta: {e}")
        return False
    
    print()
    
    # Test 2: Test chat completion semplice
    print("💬 Test 2: Chat completion semplice")
    try:
        # Test con modello diverso prima
        test_models = [
            "meta-llama/llama-3.1-8b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemma-2-9b-it:free"
        ]
        
        for test_model in test_models:
            print(f"🧪 Testing model: {test_model}")
            payload = {
                "model": test_model,
                "messages": [
                    {"role": "user", "content": "Say hello in one word"}
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
        
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"  ✅ Risposta ricevuta: '{content}'")
                return True
            else:
                try:
                    error_data = response.json()
                    print(f"  ❌ Errore: {error_data}")
                except:
                    print(f"  ❌ Response text: {response.text[:200]}")
            print()  # Spazio tra i test
        
        return False
            
    except Exception as e:
        print(f"❌ ERRORE nella chat completion: {e}")
        return False

def test_config_loading():
    """Test specifico del caricamento configurazione"""
    print("\n🔧 Test Caricamento Configurazione")
    print("=" * 50)
    
    try:
        config = get_config()
        
        print(f"Environment: {config.get_environment()}")
        print(f"Config file path: {config.config_file_path}")
        print(f"OpenRouter API Key: {'✅ Presente' if config.get_openrouter_api_key() else '❌ Assente'}")
        print(f"OpenRouter Base URL: {config.get_openrouter_base_url()}")
        print(f"Frontend URL: {config.get_frontend_url()}")
        
        # Test lettura diretta dal file config
        print("\n📄 Contenuto sezione [openrouter]:")
        if config.config.has_section('openrouter'):
            for key, value in config.config.items('openrouter'):
                if 'key' in key.lower():
                    print(f"  {key} = {'✅ presente' if value else '❌ vuoto'}")
                else:
                    print(f"  {key} = {value}")
        else:
            print("❌ Sezione [openrouter] non trovata!")
            
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Test OpenRouter Setup")
    print("=" * 50)
    
    # Test configurazione prima
    test_config_loading()
    
    # Poi test credenziali
    if test_openrouter_credentials():
        print("\n✅ Tutti i test superati!")
    else:
        print("\n❌ Test falliti - controlla la configurazione")
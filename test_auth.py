import requests
from config_manager import get_config

# Load config using config manager
config = get_config()
api_key = config.get_openrouter_api_key()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Test a simple models endpoint first
try:
    response = requests.get(
        config.get_openrouter_base_url() + '/models',
        headers=headers,
        timeout=config.get_openrouter_timeout()
    )
    print('Models endpoint status:', response.status_code)
    if response.status_code != 200:
        print('Models error:', response.text)
    else:
        print('Models endpoint works!')
        
except Exception as e:
    print('Models request failed:', str(e))

# Now test chat completion using configured model
payload = {
    'model': config.get_model('alpaca'),
    'messages': [{'role': 'user', 'content': 'Hello'}],
    'max_tokens': 50
}

try:
    response = requests.post(
        config.get_openrouter_base_url() + '/chat/completions',
        headers=headers,
        json=payload,
        timeout=config.get_openrouter_timeout()
    )
    print('Chat endpoint status:', response.status_code)
    if response.status_code != 200:
        print('Chat error:', response.text)
    else:
        print('Chat endpoint works!')
        
except Exception as e:
    print('Chat request failed:', str(e))
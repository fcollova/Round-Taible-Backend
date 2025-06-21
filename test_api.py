import requests
from config_manager import get_config

# Load config using config manager
config = get_config()
api_key = config.get_openrouter_api_key()

print('API Key loaded:', api_key[:10] + '...' if api_key else 'No API key')
print('API Key length:', len(api_key) if api_key else 0)

# Test basic OpenRouter connection
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'http://localhost:3000',
    'X-Title': 'Round TAIble'
}

# Test with a simple request using configured model
payload = {
    'model': config.get_model('mistral'),
    'messages': [{'role': 'user', 'content': 'Hello, test message'}],
    'max_tokens': 100
}

try:
    response = requests.post(
        config.get_openrouter_base_url() + '/chat/completions',
        headers=headers,
        json=payload,
        timeout=config.get_openrouter_timeout()
    )
    print('Status Code:', response.status_code)
    print('Response Headers:', dict(response.headers))
    
    if response.status_code != 200:
        print('Error response:', response.text)
    else:
        print('Success!')
        result = response.json()
        print('Response content:', result.get('choices', [{}])[0].get('message', {}).get('content', 'No content'))
        
except Exception as e:
    print('Request failed:', str(e))
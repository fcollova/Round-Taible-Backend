import requests

# Test with the exact headers that OpenRouter expects
api_key = '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'

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Test a simple models endpoint first
try:
    response = requests.get(
        'https://openrouter.ai/api/v1/models',
        headers=headers,
        timeout=10
    )
    print('Models endpoint status:', response.status_code)
    if response.status_code != 200:
        print('Models error:', response.text)
    else:
        print('Models endpoint works!')
        
except Exception as e:
    print('Models request failed:', str(e))

# Now test chat completion
payload = {
    'model': 'meta-llama/llama-3.2-3b-instruct:free',
    'messages': [{'role': 'user', 'content': 'Hello'}],
    'max_tokens': 50
}

try:
    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json=payload,
        timeout=30
    )
    print('Chat endpoint status:', response.status_code)
    if response.status_code != 200:
        print('Chat error:', response.text)
    else:
        print('Chat endpoint works!')
        
except Exception as e:
    print('Chat request failed:', str(e))
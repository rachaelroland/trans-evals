"""Test which models are available on OpenRouter."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Please set OPENROUTER_API_KEY")
    exit(1)

# Get list of available models
response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code == 200:
    models = response.json().get("data", [])
    print(f"Found {len(models)} models\n")
    
    # Show some popular ones
    popular_prefixes = ["openai/", "anthropic/", "meta-llama/", "mistralai/", "google/"]
    
    for prefix in popular_prefixes:
        print(f"\n{prefix} models:")
        prefix_models = [m for m in models if m["id"].startswith(prefix)]
        for model in prefix_models[:5]:  # Show first 5 of each type
            print(f"  - {model['id']}")
            
    # Look for gpt-3.5-turbo specifically
    print("\n\nSearching for gpt-3.5-turbo variants:")
    gpt35_models = [m for m in models if "gpt-3.5" in m["id"] or "gpt-35" in m["id"]]
    for model in gpt35_models:
        print(f"  - {model['id']}")
        
else:
    print(f"Error: {response.status_code} - {response.text}")
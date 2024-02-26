import json
import requests
import os

from dotenv import load_dotenv

# Get API key
load_dotenv()

x_api_key = os.getenv("X_API_KEY")


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': x_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()

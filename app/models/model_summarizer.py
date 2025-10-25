#import some LLM here
import base64
import requests
import os
import sys

# Claude API endpoint
API_URL = "https://api.anthropic.com/v1/messages"
API_KEY = os.getenv("CLAUDE_KEY")

def summarize_image(image_path: str):
    # Encode the image as base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Create the message for Claude
    payload = {
        "model": "claude-3-opus-20240229",  # or claude-3-sonnet if you prefer faster inference
        "max_tokens": 300,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please summarize what is shown in this image."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",  # change to image/png if needed
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    # Send the request
    response = requests.post(API_URL, headers=headers, json=payload)

    # Handle response
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None

    data = response.json()
    summary = data["content"][0]["text"]
    return summary
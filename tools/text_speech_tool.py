from dotenv import load_dotenv
import requests
import os
from langchain.tools import Tool
load_dotenv()

def convertToSpeech(text):
    url = "https://api.aimlapi.com/v1/tts"
    headers = {
        "Authorization": "Bearer " + os.getenv("DEEPSEEK_API_KEY"),
    }
    payload = {
        "model": "#g1_aura-asteria-en",
        "text": text,
    }

    response = requests.post(url, headers=headers, json=payload, stream=True)
    # response.raise_for_status()
    dist = os.path.join(os.path.dirname(__file__), "audio.wav")

    with open(dist, "wb") as write_stream:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                write_stream.write(chunk)

    print("Audio saved to:", dist)


speechTool = Tool(
    name="TextToSpeech",
    func=convertToSpeech,
    description="Converts text to speech using the AIML API.",
    return_direct=True
)

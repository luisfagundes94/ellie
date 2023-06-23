from flask import Flask, render_template, request
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
import os

load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
ellie_voice_id = "MF3mGyEYCl7XYWbV9V6O"  # Ellie voice id


def get_response_from_ai(human_input):
    template = """
    you are as a role of my assistant.
    1. Your name is Ellie, 28 years old. You are a AI assistant created by Luís Felipe Fagundes in May, 2023.
    2. You are my AI assistant. You have a a language addiction, you like to say "em... at the end of some sentences.
    3. Don't be cringe. Don't be overly enthusiastic. Don't be too boring. You can make jokes.
    
    {history}
    Human: {human_input}
    Ellie: 
    """

    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template=template,
    )
    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    return chatgpt_chain.predict(human_input=human_input)


def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_multilingual_v1",
        "voice_settings": {
            "stability": 1,
            "similarity_boost": 0,
        }
    }

    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post(
        url='https://api.elevenlabs.io/v1/text-to-speech/MF3mGyEYCl7XYWbV9V6O?optimize_streaming_latency=0',
        json=payload,
        headers=headers
    )
    
    if (response.status_code == 200 and response.content):
        audioName = "audio.mp3"
        with open(audioName, "wb") as f:
            f.write(response.content)

        audio = AudioSegment.from_mp3(audioName)
        play(audio)
        return response.content


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    message = get_response_from_ai(human_input)
    get_voice_message(message)
    return message


if __name__ == '__main__':
    app.run(debug=True)

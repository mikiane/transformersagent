import IPython
import soundfile as sf

transformers_version = "v4.29.0"
print(f"Setting up everything with transformers version {transformers_version}")

def play_audio(audio):
    sf.write("speech_converted.wav", audio.numpy(), samplerate=16000)
    return IPython.display.Audio("speech_converted.wav")

agent_name = "OpenAI (API Key)"  # change as needed

import getpass
from transformers.tools import HfAgent, OpenAiAgent

if agent_name == "StarCoder (HF Token)":
    agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
    print("StarCoder is initialized 💪")
elif agent_name == "OpenAssistant (HF Token)":
    agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
    print("OpenAssistant is initialized 💪")
elif agent_name == "OpenAI (API Key)":
    pswd = "YOUR OPENAI API KEY"
    agent = OpenAiAgent(model="text-davinci-003", api_key=pswd)
    print("OpenAI is initialized 💪")

def process_instruction(agent):
    while True:
        instruction = input("Please enter your instruction: ")
        if instruction.lower() == 'quit':
            break
        response = agent.chat(instruction, remote=True)
        print(response)

# Utilisation de la fonction
process_instruction(agent)


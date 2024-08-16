# import packages
import gradio as gr
import copy
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import shutil
import os
from chromadb.utils import embedding_functions
import gradio as gr
from PIL import Image
import requests
from io import BytesIO
from transformers import pipeline
from bark import SAMPLE_RATE, generate_audio, preload_models

# Initialize the Llama model
llm = Llama(
    ## original model
    # model_path=hf_hub_download(
    #     repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    #     filename="Phi-3-mini-4k-instruct-q4.gguf",
    # ),
    ## compressed model
    model_path=hf_hub_download(
        repo_id="TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
        filename="capybarahermes-2.5-mistral-7b.Q2_K.gguf",
    ),
    n_ctx=2048,
    n_gpu_layers=50,  # Adjust based on your VRAM
)

# use of clip model for embedding
client = chromadb.PersistentClient(path="DB")

embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader() # must be if you reads from URIs

# initialize separate collection for image and text data
def create_collection(name_image_collection,name_text_collection):
    collection_images = client.create_collection(
        name=name_image_collection,
        embedding_function=embedding_function,
        data_loader=image_loader)
    
    collection_text = client.create_collection(
        name=name_text_collection,
        embedding_function=embedding_function,
        )
    return collection_images, collection_text
collection_images,collection_text = create_collection(name_image_collection = "collection_images",name_text_collection = "collection_text")
# Get the uris to the images
IMAGE_FOLDER = '/images'  # path to image folder


image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if not image_name.endswith('.txt')])
ids = [str(i) for i in range(len(image_uris))]

collection_images.add(ids=ids, uris=image_uris)

default_ef = embedding_functions.DefaultEmbeddingFunction()
TEXT_FOLDER = "/texts" # path to text folder
text_pth = sorted([os.path.join(TEXT_FOLDER, image_name) for image_name in os.listdir(TEXT_FOLDER) if image_name.endswith('.txt')])

list_of_text = []
for text in text_pth:
    with open(text, 'r') as f:
        text = f.read()
        list_of_text.append(text)

ids_txt_list = ['id'+str(i) for i in range(len(list_of_text))]

collection_text.add(
    documents = list_of_text,
    ids =ids_txt_list
)

# Initialize the transcriber
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# Preload TTS models
preload_models()

image_path = "/dom_bremen.jpg"  # path to background image
absolute_path = os.path.abspath(image_path)

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y})["text"]

fixed_prompt = "en_speaker_5"

def generate_audio_output(text):
    audio_arr = generate_audio(text, history_prompt=fixed_prompt)
    audio_arr = (audio_arr * 32767).astype(np.int16)
    return (SAMPLE_RATE, audio_arr)
# Function to retrieve and generate text based on input query
def generate_text(message, max_tokens=150, temperature=0.2, top_p=0.9):
    try:
        # Retrieve context and image from vector store
        retrieved_image = collection_images.query(query_texts=message, include=['data'], n_results=1)
        context_text = collection_text.query(query_texts=message, n_results=1)

        context = context_text['documents'][0] if context_text else "No relevant context found."
        image_data = retrieved_image['uris'][0] if retrieved_image else None
        image_url = image_data if image_data else None

        # Log the image URL for debugging
        print(f"Retrieved image URL: {image_url}")

        # Create prompt template for LLM
        prompt_template = (
            f"Context: {context}\n\n"
            f"Question: {message}\n\n"
            f"You are a guide to city of Bremen from Germany, generate response based on context."
        )

        # Generate text using the language model
        output = llm(
                prompt_template,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repeat_penalty=1.1,
                max_tokens=max_tokens,
            )

        # Process the output
        input_string = output['choices'][0]['text'].strip()
        cleaned_text = input_string.strip("[]'").replace('\\n', '\n')
        continuous_text = '\n'.join(cleaned_text.split('\n'))

        return continuous_text, image_url[0]
    except Exception as e:
        return f"Error: {str(e)}", None

# Function to load and display an image from a file path
def load_image_from_path(file_path):
    try:
        img = Image.open(file_path)
        return img
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None
        
def process_audio(audio):
    # Transcribe the audio
    transcribed_text = transcribe(audio)
    text_output, image_path = generate_text(transcribed_text)
    if image_path:
        image_output = load_image_from_path(image_path)
    else:
        image_output = None  # Handle cases where no image is retrieved
    # return text_output, image_output
    # Generate audio output
    audio_output = generate_audio_output(text_output)
    return text_output,audio_output,image_output

def gen_tts(text):
    audio_arr = generate_audio(text, history_prompt=fixed_prompt)
    audio_arr = (audio_arr * 32767).astype(np.int16)
    return (SAMPLE_RATE, audio_arr)

# Define the Gradio interface
# with gr.Blocks() as app:
demo = gr.Interface(
        fn=process_audio,
        inputs=gr.Audio(sources=["microphone"], label="Input Audio"),
        outputs=[
            gr.Textbox(label="Generated Text"),
            gr.Audio(label="Generated Audio"),
            gr.Image(label="Retrieved Image")  # New output component for the image
        ],
        title="moinBremen - Your Personal Tour Guide for our City of Bremen",
        description="Ask your question about Bremen by speaking into the microphone. The system will transcribe your question, generate a response, and read it out loud.",
        css=""".gradio-container {
        background: url('file=/content/dom_bremen.jpg') no-repeat center center fixed;
        background-size: cover;
        }""",
        cache_examples=False,
    )

demo.launch(allowed_paths=[absolute_path])



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
collection_images = client.create_collection(
    name='collection_images',
    embedding_function=embedding_function,
    data_loader=image_loader)

collection_text = client.create_collection(
    name='collection_text',
    embedding_function=embedding_function,
    )

# Get the uris to the images
IMAGE_FOLDER = '/images' # path to image folder


image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if not image_name.endswith('.txt')])
ids = [str(i) for i in range(len(image_uris))]

collection_images.add(ids=ids, uris=image_uris)

default_ef = embedding_functions.DefaultEmbeddingFunction()
TEXT_FOLDER = "/text" # path to text folder
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

# Gradio interface function
def interface_function(message):
    text_output, image_path = generate_text(message)
    if image_path:
        image_output = load_image_from_path(image_path)
    else:
        image_output = None  # Handle cases where no image is retrieved
    return text_output, image_output

image_path = "/dom_bremen.jpg" # background image
absolute_path = os.path.abspath(image_path)

# Define the Gradio interface
iface = gr.Interface(
    fn=interface_function,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=[
        gr.Textbox(label="Generated Text"),
        gr.Image(label="Retrieved Image")
    ],
    title="Multimodal RAG Model",
    description="This interface retrieves context from a ChromaDB and generates text using an LLM model based on the retrieved context.",
    css=""".gradio-container {
        background: url('file=/content/dom_bremen.jpg') no-repeat center center fixed;
        background-size: cover;
    }"""
)

# Launch the interface
iface.launch(allowed_paths=[absolute_path])


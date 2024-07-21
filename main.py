import gradio as gr
import numpy as np
import torch
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline  # Import pipeline for ASR

# Check if CUDA (GPU) is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure your GPU is correctly set up.")

# Initialize the Llama model with GPU settings
llm = Llama(
    model_path=hf_hub_download(
        repo_id="TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
        filename="capybarahermes-2.5-mistral-7b.Q2_K.gguf",
    ),
    n_ctx=2048,
    n_gpu_layers=50,  # Adjust this value based on your GPU's VRAM
    device_map="auto"  # This will automatically choose the GPU if available
)

# Initialize the transcriber
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=0)

class VectorStore:
    def __init__(self, collection_name):
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1', device = 0)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def populate_vectors(self, texts):
        for i, text in enumerate(texts):
            embeddings = self.embedding_model.encode(text).tolist()
            self.collection.add(embeddings=[embeddings], documents=[text], ids=[str(i)])

    def search_context(self, query, n_results=1):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=n_results)
        return results['documents']

# Example initialization

# Load the plain text file
def load_plain_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return texts

texts = load_plain_text("data\data_bremen.txt")

vector_store = VectorStore("embedding_vector")
vector_store.populate_vectors(texts)

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # Use the transcriber pipeline
    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def generate_text(message, max_tokens=600, temperature=0.3, top_p=0.95):
    # Retrieve context from vector store
    context_results = vector_store.search_context(message, n_results=1)
    context = context_results[0] if context_results else ""

    # Create the prompt template
    prompt_template = (
        f"SYSTEM: You are a tour guide for the city of Bremen\n"
        f"SYSTEM: {context}\n"
        f"USER: {message}\n"
        f"ASSISTANT:\n"
    )

    # Generate text using the language model
    output = llm(
            prompt_template,
            temperature=temperature,
            top_p=top_p,
            top_k=40,
            repeat_penalty=1.1,
            max_tokens=max_tokens,
        )

    # Process the output
    input_string = output['choices'][0]['text'].strip()
    cleaned_text = input_string.strip("[]'").replace('\\n', '\n')
    continuous_text = '\n'.join(cleaned_text.split('\n'))
    return continuous_text

def process_audio(audio):
    # Transcribe the audio
    transcribed_text = transcribe(audio)
    # Generate text based on the transcribed audio
    generated_text = generate_text(transcribed_text)
    return generated_text

# Define the Gradio interface
demo = gr.Interface(
fn=process_audio,
    inputs=gr.Audio(sources=["microphone"],label="Input Audio"),
    outputs=gr.Textbox(label="Generated Text"),
    title="moinBremen - Your Personal Tour Guide for our City of Bremen",
    description="Ask your question about Bremen by speaking into the microphone. The system will transcribe your question and provide a response.",
    # Temporarily remove examples to avoid file path issues
    # examples=[
    #     ["Who is Roland ?"],
    #     ["Is Bremerhaven a part of Bremen?"],
    #     ["What is Ratskellar?"],
    #     ["What beers are produced in Bremen?"]
    # ],
    cache_examples=False,
)

# Define a function to restart the interface
def restart_interface():
    # This function can include any logic needed to reset the app's state
    return gr.update()

# Add a custom button to restart the interface
with gr.Blocks() as app:
    with gr.Row():
        demo.render()
        gr.Button("Restart Space").click(fn=restart_interface, inputs=[], outputs=[demo])

if __name__ == "__main__":
    demo.launch()

import os
import gradio as gr
import numpy as np
import torch
import faiss
import json
import codecs
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bark import SAMPLE_RATE, generate_audio, preload_models
from langchain_community.embeddings.openai import OpenAIEmbeddings  # Assuming you used OpenAI before
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
    n_gpu_layers=-1,  # Adjust this value based on your GPU's VRAM
    device_map="auto"  # This will automatically choose the GPU if available
)

# Initialize the transcriber
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=0)

# Preload TTS models
preload_models()

# Path to your local text file
file_path = r"data\data_bremen.txt"
#file_path = r"cleared_data.txt"

class VectorStore:
    def __init__(self, name, embedding_model_name="BAAI/bge-base-en-v1.5", embedding_dim=768):
        self.name = name
        self.index = faiss.IndexFlatL2(embedding_dim)  # 768 is the dimension of BAAI/bge-base-en-v1.5 embeddings
        self.collection = {}
        self.embedding_dim = embedding_dim
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def populate_vectors(self, file_path):
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create Document
        doc = Document(page_content=text)
        
        # Split Document
        splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
        chunked_docs = splitter.split_documents([doc])
        
        # Embed and add to FAISS
        for i, chunk in enumerate(chunked_docs):
            embedding = self.embedding_model.embed_documents([chunk.page_content])
            embedding_np = np.array(embedding).astype(np.float32)  # Convert to numpy array with correct dtype
            print(f"Adding embedding {i}: {embedding_np.shape}")
            self.index.add(embedding_np)
            self.collection[str(i)] = chunk.page_content

    def save(self, faiss_path, collection_path):
        faiss.write_index(self.index, faiss_path)
        with open(collection_path, 'w', encoding='utf-8') as f:
            json.dump(self.collection, f)

    def load(self, faiss_path, collection_path):
        self.index = faiss.read_index(faiss_path)
        with open(collection_path, 'r', encoding='utf-8') as f:
            self.collection = json.load(f)
        print(f"Loaded collection: {self.collection}")

    def search_context(self, query, n_results=1):
        query_embedding = self.embedding_model.embed_documents([query])
        query_embedding_np = np.array(query_embedding).astype(np.float32)
        print(f"Query embedding: {query_embedding_np.shape}")
        # Ensure the query embedding has the correct shape
        if len(query_embedding_np.shape) != 2 or query_embedding_np.shape[1] != self.embedding_dim:
            raise ValueError(f"Query embedding dimension {query_embedding_np.shape} does not match index dimension {self.embedding_dim}")
        D, I = self.index.search(query_embedding_np, n_results)
        results = [self.collection[str(idx)] for idx in I[0]]
        print(f"Search results: {results}")
        return results

# Paths to your files
faiss_path = r"data\faiss_index2.bin"
collection_path = r"data\collection2.json"

# Check if the files exist
if not os.path.exists(faiss_path) or not os.path.exists(collection_path):
    raise FileNotFoundError(f"One or both files {faiss_path}, {collection_path} do not exist.")

# Load the precomputed FAISS index and collection
vector_store = VectorStore("embedding_vector")
vector_store.load(faiss_path, collection_path)

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def generate_text(message, max_tokens=150, temperature=0.2, top_p=0.9):
    # Retrieve context from vector store
    context_results = vector_store.search_context(message, n_results=1)
    context = context_results[0] if context_results else ""

    # Create the prompt template
    # Create the prompt template
    prompt_template = (
        f"Context: {context}\n\n"
        f"Question: {message}\n\n"
        f"Based on the provided context, give a concise and accurate answer to the question."
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
    return continuous_text

fixed_prompt = "en_speaker_5"

def generate_audio_output(text):
    audio_arr = generate_audio(text, history_prompt=fixed_prompt)
    audio_arr = (audio_arr * 32767).astype(np.int16)
    return (SAMPLE_RATE, audio_arr)

def process_audio(audio):
    # Transcribe the audio
    transcribed_text = transcribe(audio)
    print(f"Transcribed text: {transcribed_text}")
    # Generate text based on the transcribed audio
    generated_text = generate_text(transcribed_text)
    print(f"Generated text: {generated_text}")
    # Generate audio output
    audio_output = generate_audio_output(generated_text)
    return generated_text, audio_output


def gen_tts(text):
    audio_arr = generate_audio(text, history_prompt=fixed_prompt)
    audio_arr = (audio_arr * 32767).astype(np.int16)
    return (SAMPLE_RATE, audio_arr)

# Define the Gradio interface
with gr.Blocks() as app:
    demo = gr.Interface(
        fn=process_audio,
        inputs=gr.Audio(sources=["microphone"], label="Input Audio"),
        outputs=[gr.Textbox(label="Generated Text"), gr.Audio(label="Generated Audio")],
        title="moinBremen - Your Personal Tour Guide for our City of Bremen",
        description="Ask your question about Bremen by speaking into the microphone. The system will transcribe your question, generate a response, and read it out loud.",
        cache_examples=False,
    )


if __name__ == "__main__":
    app.queue()
    app.launch()

import gradio as gr
import numpy as np
import torch
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bark import SAMPLE_RATE, generate_audio, preload_models

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
    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def generate_text(message, max_tokens=150, temperature=0.2, top_p=0.9):
    # Retrieve context from vector store
    context_results = vector_store.search_context(message, n_results=1)
    context = context_results[0] if context_results else ""

    # Create the prompt template
    prompt_template = (
      f"SYSTEM: You are a tour guide for the city of Bremen answering the question. \n"
      f"SYSTEM: {context}\n"
      f"USER: {message}\n"
      f"ASSISTANT: Here are some examples of short answers: (e.g., The answer is..., In short, ...)\n"
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
    # Generate text based on the transcribed audio
    generated_text = generate_text(transcribed_text)
    # output = pipe(generated_text)
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

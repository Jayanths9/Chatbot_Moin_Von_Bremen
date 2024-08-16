# 🌟 Moin Von Bremen 🌟

Welcome to "Moin von Bremen," a fun and interactive project developed by a group of enthusiastic Master's students from the University of Bremen. In this project, we dive into the world of LLMs (Large Language Models), explore the power of Retrieval Augmented Generation (RAG), and experiment with the concept of multimodality. Together, we created an audio city guide for our beloved city of Bremen! 🎧🗺️

## 🚀 Project Overview

This project is a fascinating journey that starts with the idea of building a chatbot capable of serving as an audio city guide. What better city to choose than Bremen, our home while studying at the University of Bremen? With this guide, you'll get to know Bremen like never before!

### How It Works

We combined our local knowledge with reliable facts from Wikipedia to create an engaging and informative experience. Here's a step-by-step breakdown of how we did it:

1. **Data Generation**: We sourced images and data from Wikipedia to support our chatbot. Check out the `datageneration.ipynb` file for more details on how this was done.

2. **LLM & RAG**: Curious about building your very own personal bot? So were we! We delved into LLMs and the ever-popular RAG technique to develop a domain-specific knowledge application. RAG is widely used by large businesses to create specialized applications. Want to learn more? Read this insightful [RAG Article](#) 📖.

3. **Text Embeddings with ChromaDB**: Our journey continued with the `textdata_chromadb.py` file, where we developed RAG using vector embeddings with ChromaDB. We even built an API using Gradio for a smooth user interface. We also experimented with multimodal concepts by creating collections for both text and images. When given a prompt, the system searches for the most relevant image and text in the database.

4. **Vector Embedding**: The concept of vector embedding extends to creating relevant numeric contexts, which are used during searches to pull the most relevant data from the database. For an in-depth explanation, check out this [article on Embeddings and Vector Databases](https://medium.com/@vladris/embeddings-and-vector-databases-732f9927b377).

5. **Audio Guide with FAISS**: Moving forward, we developed an audio guide by implementing the code in `audiodata_faissEmbedding.py`. We used OpenAI’s Whisper ASR model for audio-to-text conversion. Learn more about Whisper [here](https://openai.com/index/whisper/). For an interesting deep dive into mel spectrograms, check out this [article](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53).

6. **Similarity Search with FAISS**: FAISS, developed by Facebook AI Research, is designed for efficient similarity search and clustering of dense vectors. It's incredibly useful for finding similar items in a dataset based on their vector representations. For a detailed explanation, visit the [FAISS official page](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/).

### ChromaDB vs. FAISS

Why did we choose ChromaDB over FAISS for this project? Here's a quick comparison:

- **FAISS**: A specialized library for efficient similarity search, focusing primarily on handling and querying vectors.
- **ChromaDB**: A more comprehensive database system specifically designed for embeddings, with advanced features for managing collections, querying, filtering, and handling multi-modal data.

For multi-modal searches (like searching text with image embeddings), ChromaDB offers more flexibility than FAISS. We break down our decision-making process and the implementation in `Main.py`, illustrated in the following diagram (insert diagram here).


## 🛠️ Setup

To get started with this project, follow these steps:

1. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Data**: The current implementation data is present in the `data` folder.





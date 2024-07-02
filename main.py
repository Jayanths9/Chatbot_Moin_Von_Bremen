import os, logging
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

from utils.helper import print_answers

#A DocumentStore stores the Documents that the question answering system uses to find answers to your questions.
document_store = InMemoryDocumentStore(use_bm25=True)

doc_dir = "data/"
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]

indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)

while True:
  # Get user input for the question
  user_query = input("Ask your question (or 'quit' to exit): ")

  # Check if user wants to quit
  if user_query.lower() == "quit":
    break

  # Run the Haystack pipeline with user query
  prediction = pipe.run(query=user_query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})

  # Print the answers using your modified print_answers function
  print_answers(prediction, details="minimum")  # Adjust details if needed

print("Goodbye!")

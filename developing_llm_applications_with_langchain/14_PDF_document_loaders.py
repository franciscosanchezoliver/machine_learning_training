"""
PDF document loaders
--------------------
To begin implementing Retrieval Augmented Generation (RAG), you'll first need 
to load the documents that the model will access. These documents can come from 
a variety of sources, and LangChain supports document loaders for many of them.

In this exercise, you'll use a document loader to load a PDF document containing 
the paper, RAG VS Fine-Tuning: Pipelines, Tradeoffs, and a Case Study on 
Agriculture by Balaguer et al. (2024).

Note: pypdf, a dependency for loading PDF documents in LangChain.

Instructions:
- Import the appropriate class for loading PDF documents in LangChain.
- Create a document loader for the 'rag_vs_fine_tuning.pdf' document.
- Load the document into memory to view the contents of the first document, or page.
"""

# Import library
import os
from langchain_community.document_loaders import PyPDFLoader

pdf_file_path = os.path.join(
    os.getcwd(), 
    "data", 
    "rags_vs_finetuning_pipelines_tradeoff_and_case_of_study_on_agriculture.pdf"
)

loader = PyPDFLoader(
    file_path=pdf_file_path
)

# Load the document
data = loader.load()
print(data[0])
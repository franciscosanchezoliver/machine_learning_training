"""
PDF document loaders
--------------------
To begin implementing Retrieval Augmented Generation (RAG), you'll first need 
to load the documents that the model will access. These documents can come from 
a variety of sources, and LangChain supports document loaders for many of them.

In this exercise, you'll use a document loader to load a PDF document containing 
the famous paper, Attention is All You Need.

Instructions
- Import the appropriate class for loading PDF documents in LangChain.
- Create a document loader for the 'attention_is_all_you_need.pdf' document, 
  which is available in the current directory.
- Load the document into memory to view the contents of the first document, 
  or page.
"""

import os
from langchain_community.document_loaders import PyPDFLoader

pdf_file_path = os.path.join(os.getcwd(), "data", "attention_is_all_you_need.pdf")
loader = PyPDFLoader(
    file_path=pdf_file_path 
)

# Load the document
data = loader.load()

print(f"""
-----------------------
PDF Content (firt page)
-----------------------
{data[0]}

-------------
File Metadata
-------------
{data[0].metadata}
""")


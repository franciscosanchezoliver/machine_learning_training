"""
Preparing the documents and vector database
-------------------------------------------
Over the next few exercises, you'll build a full RAG workflow to have a 
conversation with a PDF document containing the paper: 

    RAG VS Fine-Tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture by 
    Balaguer et al. (2024). 

This works by splitting the documents into chunks, storing them in a vector 
database, defining a prompt to connect the retrieved documents and user input, 
and building a retrieval chain for the LLM to access this external data.

In this exercise, you'll prepare the document for storage and ingest them into 
a Chroma vector database. You'll use a RecursiveCharacterTextSplitter to chunk 
the PDF, and ingest them into a Chroma vector database using an OpenAI 
embeddings function.

Instructions:
1. Assign your OpenAI API key to openai_api_key.
2. Split the documents in data using a recursive character splitter with a 
  chunk_size of 300 and chunk_overlap of 50; leave the separators argument out, 
  as it defaults to ["\n\n", "\n", " ", ""].
3. Define an OpenAI embeddings model and use it to embed and ingest the 
  documents into a Chroma database.
4. Configure vectorstore into a retriever object that returns the top 3 documents 
  for use in the final RAG chain.
"""

import os

# Load External data
from langchain_community.document_loaders import PyPDFLoader

# Split the documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector database
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

# To create a prompt template and combine the retrieved documents chunks
# with the use input question
from langchain_core.prompts import ChatPromptTemplate

# The LLM
from langchain_openai import ChatOpenAI

from langchain_core.runnables import RunnablePassthrough


# Set your API Key from OpenAI
openai_api_key = os.environ["openai_apikey"]

pdf_file_path = os.path.join(
    os.getcwd(),
    "data",
    "rags_vs_finetuning_pipelines_tradeoff_and_case_of_study_on_agriculture.pdf",
)

# Load the PDF
loader = PyPDFLoader(file_path=pdf_file_path)
data = loader.load()

print(
    f"""
PDF Content (first page)
------------------------
{data[0]}
"""
)

# 2. Split the documents in data using a recursive character splitter with a
# chunk_size of 300 and chunk_overlap of 50; leave the separators argument out,
# as it defaults to ["\n\n", "\n", " ", ""].
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(data)
print(
    f"""
Chunks obtained using RecursiveCharacterTextSplitter
----------------------------------------------------
{len(docs)}

First 5 chunks
--------------"""
)

for index, each_doc in enumerate(docs[:5]):
    print(f"\nChunk {index + 1}")
    print(len(f"Chunk {index + 1}") * "-")
    print(each_doc.page_content)


# 3. Define an OpenAI embeddings model and use it to embed and ingest the
# documents into a Chroma database.
embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
# A Chroma vector store is created from the documents (docs). The
# OpenAIEmbeddings class is used to generate embeddings for the documents
vector_database = Chroma.from_documents(
    docs,  # Documents we want to convert to vector
    embedding=embedding_function,
    persist_directory=os.getcwd(),
)

# To integrate the database with other LangChain components, we need to
# convert the vector database into a retriever.
retriever = vector_database.as_retriever(
    search_type="similarity",  # Perform a similarity search
    search_kwargs={"k": 3},  # Return the top 3 most similar documents
)

# Building a retrieval prompt template
# ------------------------------------
# Now your documents have been ingested into vector database and are ready for
# retrieval, you'll need to design a chat prompt template to combine the retrieved
# document chunks with the user input question.

# Add placeholders to the message string
message = """
Answer the following question using the context provided:

Context:
{context}

Question:
{question}

Answer:
"""

# Create a chat prompt template from the message string
prompt_template = ChatPromptTemplate.from_messages([("human", message)])

# Creating a RAG chain
# --------------------
# Now to bring all the components together in your RAG workflow! You've prepared
# the documents and ingested them into a Chroma database for retrieval. You
# created a prompt template to include the retrieved chunks from the academic
# paper and answer questions.
# Set your API Key from OpenAI

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key
)

# Create a chain to link retriever, prompt_template, and llm
# The retriever includes the vector database with the pdf document we just
# loaded.
# The RunnablePassthrough is used to pass the user input question.
# The prompt template allow us to concatenate the retrieved document
# with the user input question
# The llm is used to generate the final answer
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
)

# Invoke the chain
response = rag_chain.invoke("Which popular LLMs were considered in the paper?")
print(response.content)

# In this exercise we created a RAG workflow to allow you to talk with a PDF

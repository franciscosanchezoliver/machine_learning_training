"""
Prompt templates and chaining
-----------------------------
In this exercise, you'll begin using two of the core components in 
LangChain: prompt templates and chains!

Prompt templates are used for creating prompts in a more modular way, so they 
can be reused and built on. 
Chains act as the glue in LangChain; bringing the other components together 
into workflows that pass inputs and outputs between the different components.

The classes necessary for completing this exercise, including 
HuggingFaceEndpoint, have been pre-loaded for you.
"""

import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Create a template for the prompt
prompt_template = """
You are an artificial intelligence assistant, answer the question. {question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["question"])

# Use a Hugging Face model using the API.
hugging_face_api_key = os.environ["huggingface_apikey"]
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=hugging_face_api_key,
)

# Chain the prompt with the LLM
chain = prompt | llm

# Ask something
output = chain.invoke(
    "How does LangChain make LLM application development easier?"
)

print(output)

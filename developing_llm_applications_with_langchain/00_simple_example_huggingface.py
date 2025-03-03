"""
Hugging Face models in LangChain
--------------------------------
There are thousands of language models freely available to use on Hugging Face. 
Hugging Face integrates really nicely into LangChain, so in this exercise, 
you'll use LangChain to load and predict using a model from Hugging Face.

To complete this exercise, you'll need first need to create a Hugging Face 
API token. Creating this token is completely free, and there are no charges 
for loading models.

Sign up for a Hugging Face account at https://huggingface.co/join
Navigate to https://huggingface.co/settings/tokens
Select "New token" and copy the key
"""


import os
from langchain_huggingface import HuggingFaceEndpoint


hugging_face_api_key = os.environ["huggingface_apikey"]

# https://huggingface.co/tiiuae/falcon-7b-instruct
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct", 
    huggingfacehub_api_token=hugging_face_api_key 
)

question = "Can you still have fun"

output = llm.invoke(question)

print("hello")
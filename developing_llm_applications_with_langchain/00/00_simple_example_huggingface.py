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

from langchain_huggingface import HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100
    }
)

print(llm.invoke("What is Hugging Face?"))

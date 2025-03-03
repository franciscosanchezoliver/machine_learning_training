import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(
    input_variables=["adjective"], template=prompt_template
)

hugging_face_api_key = os.environ["huggingface_apikey"]
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct", 
    huggingfacehub_api_token=hugging_face_api_key 
)

chain = prompt | llm

output = chain.invoke("black humor")

print(output)



import os
from langchain.chat_models import ChatOpenAI

openai_api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=openai_api_key
)

print(llm.invoke("What is langchain?"))

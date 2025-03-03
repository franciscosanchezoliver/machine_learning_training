import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


openai_apikey = os.environ["openai_apikey"]

llm = ChatOpenAI(temperature=0, openai_api_key=openai_apikey, verbose=True)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a master zombie slayer in an apocalyptic world"),
        ("human", "Respond to the question: {question}"),
    ]
)

full_prompt = prompt_template.format_messages(
    question="What is the most effective but yet easy to self build weapon I can build?"
)

llm(full_prompt)


print("hello")

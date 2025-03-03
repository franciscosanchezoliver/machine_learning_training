from langgraph.prebuilt import create_react_agent
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI
import os


openai_apikey = os.environ["openai_apikey"]

llm = ChatOpenAI(
    openai_api_key=openai_apikey
)

# Use a tool to the mathematical calculations as LLMs strugle to 
# do math 
tools = load_tools(["llm-math"], llm=llm)

agents = create_react_agent(
    model=llm, 
    tools=tools
)

messages = agents.invoke(
    {
        "messages": [
            ("human","What is the square root of 101")
        ]
    }
)

print(messages)
print(messages['messages'][-1].content)

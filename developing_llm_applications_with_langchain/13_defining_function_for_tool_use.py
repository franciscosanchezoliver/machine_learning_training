"""
Defining a function for tool use
--------------------------------
You're working for a SaaS (software as a service) company with big goals for 
rolling out tools to help employees at all levels of the organization to make 
data-informed decisions. You're creating a PoC for an application that allows 
customer success managers to interface with company data using natural language 
to retrieve important customer data.

You've been provided with a pandas DataFrame called customers that contains a 
small sample of customer data. Your first step in this project is to define a 
Python function to extract information from this table given a customer's name. 

Instructions
- Define a retrieve_customer_info() function that takes a string argument, name.
- Filter the customers DataFrame to return rows with "name" equal to the input argument, name.
- Call the function on the customer name, "Peak Performance Co.".
- Now that you have a function for extracting customer data from the customers 
  DataFrame, it's time to convert this function into a tool that's compatible 
  with LangChain agents.
- Now that you have your tools at-hand, it's time to set up your agentic 
  workflow! You'll again be using a ReAct agent, which, recall, reasons on the 
  steps it should take, and selects tools using this context and the tool 
  descriptions.
"""

import os
import pandas as pd
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent


# We can define a customer data example dataset
# Define the data
data = {
    "id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "name": [
        "Tech Innovators Inc.", "Green Solutions Ltd.", "Global Enterprises", 
        "Peak Performance Co.", "Visionary Ventures", "NextGen Technologies", 
        "Dynamic Dynamics LLC", "Infinity Services", "Eco-Friendly Products", 
        "Future Insights"
    ],
    "subscription_type": ["Premium", "Standard", "Basic", "Premium", "Standard", 
                          "Basic", "Premium", "Standard", "Basic", "Premium"],
    "active_users": [450, 300, 150, 800, 600, 200, 700, 500, 100, 900],
    "auto_renewal": [True, False, True, True, False, True, True, False, True, True]
}

# Create the DataFrame
customers = pd.DataFrame(data)

# Define a function to retrieve customer info by-name
@tool
def retrieve_customer_info(name: str) -> str:
    """Retrieve customer information based on their name."""
    # Filter customers for the customer's name
    customer_info = customers[customers['name'] == name]
    return customer_info.to_string()
  
# Check the arguments of the tool
print(retrieve_customer_info.args)

# Call the function on Peak Performance Co.
print(retrieve_customer_info("Peak Performance Co."))

openai_apikey = os.environ["openai_apikey"]

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_apikey
)

agent = create_react_agent(
    model=llm,
    tools = [retrieve_customer_info]
)

messages = agent.invoke({
    "messages": [
        ("human", "Create a summary of our customer: Peak Performance Co.")
    ]
})

print(messages)


#: [{'id': 'call_rax3qErajUYa8Jr1vtFuEqNV', 'function': {'arguments': '{"name":"Peak Performance Co."}', 'name': 'retrieve_customer_info'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 58, 'total_tokens': 76}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-75c875bd-ccbc-4327-b7c0-fe336cf6f700-0', tool_calls=[{'name': 'retrieve_customer_info', 'args': {'name': 'Peak Performance Co.'}, 'id': 'call_rax3qErajUYa8Jr1vtFuEqNV'}], usage_metadata={'input_tokens': 58, 'output_tokens': 18, 'total_tokens': 76}), ToolMessage(content='    id                  name subscription_type  active_users  auto_renewal\n3  104  Peak Performance Co.           Premium           800          True', name='retrieve_customer_info', id='ed70c3e2-6a08-4bc0-afc5-24c47c3b0eff', tool_call_id='call_rax3qErajUYa8Jr1vtFuEqNV'), AIMessage(content='Peak Performance Co. is a customer with the following details:\n- ID: 104\n- Subscription Type: Premium\n- Active Users: 800\n- Auto Renewal: True', response_metadata={'token_usage': {'completion_tokens': 38, 'prompt_tokens': 116, 'total_tokens': 154}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-fb4b0689-e550-4cb9-a62d-34f58fe4dc9e-0', usage_metadata={'input_tokens': 116, 'output_tokens': 38, 'total_tokens': 154})]}







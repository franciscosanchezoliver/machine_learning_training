import os
from langgraph.prebuilt import create_react_agent
from langchain.agents import load_tools
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Define a custom tool so our llm can use it to generate a custom financial
# report about a company
@tool
def financial_report(
    company_name: str,
    revenue: int,
    expenses: int
) -> str:
    """Generate a financial report for a company that calculates net income."""
    net_income = revenue - expenses
    report = f"Financial Report for {company_name}:\n"
    report += f"Revenue: ${revenue}\n"
    report += f"Expenses: ${expenses}\n"
    report += f"Net income: ${net_income}\n"
    return report


openai_apikey = os.environ["openai_apikey"]

llm = ChatOpenAI(
    api_key=openai_apikey,
    temperature=0
)

agent = create_react_agent(
    llm, 
    tools=[financial_report]
)

messages = agent.invoke({
    "messages": [
        ("human", "TechStack generated made $10 millions with $8 million of costs. Generate a financial report.")
    ]
})

print(messages)

# {'messages': [HumanMessage(content='TechStack generated made $10 millions with $8 million of costs. Generate a financial report.', id='ac720e58-0be7-40b3-a37a-0a190b775091'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_ycmkKHHVeOUpYQVn8BpGPS5r', 'function': {'arguments': '{"company_name":"TechStack","revenue":10000000,"expenses":8000000}', 'name': 'financial_report'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 30, 'prompt_tokens': 78, 'total_tokens': 108}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-954daf56-4ffe-4871-a0d6-cb2b75ecb134-0', tool_calls=[{'name': 'financial_report', 'args': {'company_name': 'TechStack', 'revenue': 10000000, 'expenses': 8000000}, 'id': 'call_ycmkKHHVeOUpYQVn8BpGPS5r'}], usage_metadata={'input_tokens': 78, 'output_tokens': 30, 'total_tokens': 108}), ToolMessage(content='Financial Report for TechStack:\nRevenue: $10000000\nExpenses: $8000000\nNet income: $2000000\n', name='financial_report', id='da0bfbbe-4bf8-4051-b515-c89d49b4731a', tool_call_id='call_ycmkKHHVeOUpYQVn8BpGPS5r'), AIMessage(content='Here is the financial report for TechStack:\n- Revenue: $10,000,000\n- Expenses: $8,000,000\n- Net Income: $2,000,000', response_metadata={'token_usage': {'completion_tokens': 40, 'prompt_tokens': 145, 'total_tokens': 185}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-18b8ea63-bdd6-4a6d-a3f4-9ff360ccc50f-0', usage_metadata={'input_tokens': 145, 'output_tokens': 40, 'total_tokens': 185})]}






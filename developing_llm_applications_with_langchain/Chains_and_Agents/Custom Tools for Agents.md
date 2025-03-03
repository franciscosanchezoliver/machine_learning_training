#LLM #agents #tools

Now that we've created our first agent, let's take a closer look at **tools** so **we can design our own**.

**Tools in LangChain must be formatted in a specific way to be compatible with agents**. They **must have a name**, accessible via the **.name attribute**. 

```python
from langchain.agents import load_tools

tools = load_tools(["llm-math"], llm=llm)
print(tools[0].name) 
# Output: Calculator

# A description, which is used by an LLM to determine when 
# to call the tool.
print(tools[0].description)
# Output: Useful for when you need to answer questions about match

# Finally the "return_direct" parameter indicates whether the 
# agent should stop after invoking this tool, which 
# it won't in this case.
print(tools[0].return_direct)
# Output: False
```

Understanding this required format will help us to understand how to create our own tools. 

**Let's say we want to create a function to generate a financial report for a company**

```python
def financial_report(
	company_name: str,
	revenue: int,
	expenses: int
) -> str:
	net_income = revenue - expenses

	# Return an string representing the net income
	report = f"Financial Report for {company_name}:\n"
	report += f"Revenue: ${revenue}\n"
	report += f"Expenses: ${expenses}\n"
	report += f"Net Income: ${net_income}\n"
	return report

print(financial_report(
		company_name="Nike",
		revenue=100,
		expenses=50
))

# Output: 
# Financial Report for Nike:
# Revenue: $100
# Expenses: $50
# Net Income: $50
```

**Let's convert this  a tool our agent can call**. To do this, we **import the @tool decorator and add it before the function **definition.

Like with the built-in tool we were looking at, we can now examine the various attributes of our tool. This include **its name, which is the function name by default**, **its description, which is the function's docstring**, and ***return_direct*, which is set to False by default.**

```python
@tool
def financial_report(
	company_name: str,
	revenue: int,
	expenses: int
) -> str:
	net_income = revenue - expenses

	# Return an string representing the net income
	report = f"Financial Report for {company_name}:\n"
	report += f"Revenue: ${revenue}\n"
	report += f"Expenses: ${expenses}\n"
	report += f"Net Income: ${net_income}\n"
	return report

print(financial_report.name)
# Output: financial_report

print(financial_report.description)
# Output: Generate a finantial report for a company that calculates
#         net income

print(financial_report.return_direct)
# Output: False

print(financial_report.args)
# Output: 
# {
#  "company_name": {
# 	"title": "Company Name", 
# 	"type": "string"
#  },
#  "revenue": {
# 	"title": "Revenue",
# 	"type": "integer"
#  },
#  "expenses": {
# 	"title": "Expenses",
# 	"type": "integer"
#  }
# }
```

Let's put our tool into action. **We'll use a ReAct agent, combining the chat LLM with a list of tools to use, containing our new custom tool.**

```python
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(
		api_key=openai_api_key,
		temperature=0
)
agent = create_react_agent(
			llm,
			[financial_report]
)

# We invoke the agent with an input containing the required information:
# - A company name
# - Revenue
# - Expenses
messages = agent.invoke({"messages": [("human": "TechStack generated made $10 millions with 8 millions of costs. Generate a finantial report")]})

print(messages)

# Output:
{
 'messages': [
	HumanMessage(content="TechStack generated made $10..."),
	AIMessage(
		content='',
		tool_calls=[{'name': financial_report}],
	)
 
 ]
}
```

[Example of a custom tool, financial calculator](./../12_custom_tool.py)
[Example of a custom tool, financial calculator](./../13_defining_function_for_tool_use.py)










#LLM #agents #ReAct #LangGraph

Now that we're confident with LLMs, it's time to move onto agents.

In LangChain, **agents use language models to determine actions**. **Agents** often **use tool**s, **which are functions called by the agent** to interact with the system. 

**These tools can be high-level utilities to transform inputs, or** they can be **task-specific**. 

**Agents can even use chains and other agents** as tools. 

**We'll discuss** a type of agent called **ReAct agents**.

## ReAct Agents

ReAct stands for **Reasoning and Acting**, and this is exactly how the agent operates. It prompts the model using a **repeated loop of thinking, acting, and observing**.

**If we were to ask a ReAct agent that had access to a weather tool, "What is the weather like in Kingston**, Jamaica?", it would **start by thinking about the task and which tool to call,** **call that tool using information**, **and observe the result from the tool call**.

```
Though: I should call Weather() to find the weather in Kingston, Jamaica.

Act: Weather("Kingston, Jamaica")

Observe: The weather is mostly sunny with temperatures of 82ºF
```

**To implement agents we'll be using LangGraph**, which is a branch of the LangChain ecosystem **specifically for designing agentic systems**, or systems including agents. 
Like LangChain core library, it is built to provide an unified, tool-agnostic syntax.

We'll be using the following version for this course: 
```python
pip install langgraph==0.0.66
```


We'll create a ReAct agent that can solve match problems (something most LLMs struggle with).

```python
from langgraph.prebuilt import create_react_agent
from langchain.agents import load_tools

llm = ChatOpenAI(openai_api_key=openai_api_key)
tools = load_tools(["llm-math"], llm=llm)
agent = create_react_agent(llm, tools)

messages = agent.invoke({"messages": [("human": "What is the square root of 101")]})
print(messages)

# {
# 'messages': [
#	HumanMessage(content='What is the square root of 101?'),
#	AIMessage(
#		content='', 
#		..., 
#		tool_calls=[
#			{'name':'Calculator'},
#			'args': {'__arg1':'sqrt(101)'}, 
#		]
#	),
#	ToolMessage(content='Answer: 10.049875', ...),
#	AIMessage(content='The square root of 101 is approximately 10.05', ...)
# ]
#}

# To print the final response
print(messages['messages'][-1].content)
# The square root of 101 is approximately 10.05

```

There is a lot of metadata in the output, so we've trimmed it for brevity.

We can see that executing the agent resulted in a series of messages. 
- The first is our prompt defining the problem
- The second is created by the model to identify the tool to use and to convert our query into mathematical format. 
- The third  is the result of the tool call.
- The final message is the model's response after observing the tool's answer, which it decided to round to two decimal places.

### Exercise: What's an agent?
Which are the following statements about agents are correct?

Select all correct answers:
-  [x] Agents use language models to make decisions
-  [ ] Agents store external data for retrieval
-  [x] Agents use tools to interact with the system in different ways
-  [ ] Agents are used to create reusable and customizable prompts


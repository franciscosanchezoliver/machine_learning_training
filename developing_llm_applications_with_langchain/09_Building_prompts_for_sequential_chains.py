"""
Building prompts for sequential chains
--------------------------------------
Over the next couple of exercises, you'll work to create a system for helping 
people learn new skills. This system needs to be built sequentially, so learners 
can modify plans based on their preferences and constraints. 

You'll utilize your LangChain LCEL skills to build a sequential chain to build 
this system, and the first step is to design the prompt templates that will 
be used by this system.

Instructions:
- Create a prompt template called learning_prompt that takes an input 
  "activity" and creates a learning plan.
- Create a prompt template called learning_prompt that takes an input 
  "learning_plan" and modifies it to fit within one week.
- Invoke the learning_prompt with an activity of your choice (try "play golf" 
  if you're struggling for ideas).
"""

from langchain_core.prompts import PromptTemplate


# Create a prompt template that takes an input activity
learning_prompt = PromptTemplate(
    input_variables=["activity"],
    template="I want to learn how to {activity}. Can you suggest how I can learn this step-by-step?"
)

# Create a prompt template that places a time constraint on the output
time_prompt = PromptTemplate(
    input_variables=["learning_plan"],
    template="I only have one week. Can you create a plan to help me hit this goal: {learning_plan}."
)

# Chain the prompts together using LCEL

# Invoke the learning_prompt with an activity
print(
    learning_prompt.invoke({"activity": "create a machine learning program"})
)











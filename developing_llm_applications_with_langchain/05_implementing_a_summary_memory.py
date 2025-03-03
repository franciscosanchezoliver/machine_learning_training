"""
Implementing a summary memory
-----------------------------

For longer conversations, storing the entire memory, or even a long buffer 
memory, may not be technically feasible. In these cases, a summary memory 
implementation can be a good option. 

Summary memories summarize the conversation at each step to retain the key 
context for the model to use. This works by using another LLM for generating 
the summaries, alongside the LLM used for generating the responses.

In this exercise, you'll implement a chatbot summary memory, using an OpenAI 
model for generating the summaries.

All of the LangChain classes necessary for completing this exercise have 
been pre-loaded for you.

Instructions:
- Define a summary memory that uses the same OpenAI model for generating 
  the summaries.
- Define a conversation chain for integrating the model and summary memory
- Invoke the chain with the inputs provided.
"""

import os
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAI
from langchain.chains import ConversationChain

openai_api_key = os.environ["openai_apikey"]

chat = OpenAI(
    model_name="gpt-3.5-turbo-instruct", 
    temperature=0, 
    openai_api_key=openai_api_key
)

# User another LLM to do the summary.
memory = ConversationSummaryMemory(
    llm=OpenAI(
        model_name="gpt-3.5-turbo-instruct",
        openai_api_key=openai_api_key
    ),
)

conversation_chain = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

# Let's ask the model a few questions to see how it is using the memory
conversation_chain.predict(
    input="Describe the relationship of the human mind with the keyboard when "
          "taking a great online class."
)

conversation_chain.predict(
    input="Use an analogy to describe it"
)

# Output:

# Entering new ConversationChain chain...
# Prompt after formatting:
# following is a friendly conversation between a human and an AI. The AI is 
# talkative and provides lots of specific details from its context. If the AI 
# does not know the answer to a question, it truthfully says it does not know.
# 
# Current conversation:
# The human and AI discuss the relationship between the human mind and the 
# keyboard in a great online class, noting that it is one of collaboration 
# and communication. The keyboard acts as a tool for the human mind to interact 
# with the online class, allowing for a seamless flow of information and 
# learning. It also allows for the expression of thoughts and ideas in a 
# written format, enhancing the learning experience.
# Human: Use an analogy to describe it
# AI:



# Note: 
# Like when choosing models for generating responses, you may need to experiment 
# with different summary models to find the best one for your use case. The 
# Hugging Face Hub has models specifically trained for summarization, so this 
# is often a good place to start.

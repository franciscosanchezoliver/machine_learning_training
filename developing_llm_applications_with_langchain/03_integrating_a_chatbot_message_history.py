"""
Integrating a chatbot message history
-------------------------------------

A key feature of chatbot applications is the ability to have a conversation, 
where context from the conversation is stored and available for the model 
to access.

In this exercise, you'll create a conversation history that will be passed 
to the model. This history will contain every message in the conversation, 
including the user inputs and model responses.

All of the LangChain classes necessary for completing this exercise have 
been pre-loaded for you.

Instructions:
- Create a conversation history and add the first AI message.
- Add the user message and call the model on the messages in the 
  conversation history.
- Add another user message and call the model on the updated message history.

"""

import os
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory

chat = ChatOpenAI(
    temperature=0, 
    openai_api_key=os.environ["openai_apikey"] 
  )

# Create the conversation history and add the first AI message
history = ChatMessageHistory()
history.add_ai_message("Hello! Ask me anything about Python programming!")

# Add the user message to the history and call the model
history.add_user_message("What is a list comprehension?")
ai_response = chat(history.messages)
print(ai_response)

# Add another user message and call the model
history.add_user_message("Describe the same in fewer words")
ai_response = chat(history.messages)
print(ai_response)

# Output:
#
# A list comprehension is a concise way to create lists in Python. It allows you 
# to generate a new list by applying an expression to each item in an existing 
# iterable ...' 
# response_metadata={
#   'token_usage': {
#     'completion_tokens': 165, 
#     'prompt_tokens': 26, 
#     'total_tokens': 191
#   }, 
#   'model_name': 'gpt-3.5-turbo', 
#   'system_fingerprint': None, 
#   'finish_reason': 'stop', 
#   'logprobs': None
# } id='run-c8b71de5-66ab-4072-9cd0-28c85c8b935a-0'
#
# content='A concise way to create lists in Python using a single line of code.' 
# response_metadata={
#   'token_usage': {'completion_tokens': 15, 'prompt_tokens': 36, 'total_tokens': 51}, 
#   'model_name': 'gpt-3.5-turbo', 
#   'system_fingerprint': None, 
#   'finish_reason': 'stop', 
#   'logprobs': None} id='run-0bd22457-32d9-4f39-839f-1579b0d334e3-0'


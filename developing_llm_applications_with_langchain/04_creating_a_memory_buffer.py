"""
Creating a memory buffer
------------------------

For many applications, storing and accessing the entire conversation history 
isn't technically feasible. In these cases, the messages must be condensed 
while retaining as much relevant context as possible. 

One common way of doing this is with a memory buffer, which stores only the 
most recent messages.

In this exercise, you'll integrate a memory buffer into an OpenAI chat model 
using a chain.

Instructions
- Define a buffer memory that stores the four most recent messages.
- Define a conversation chain for integrating the model and memory buffer
- Invoke the chain with the inputs provided.
"""

import os
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


openai_api_key = os.environ["openai_apikey"]

chat = OpenAI(
    model_name="gpt-3.5-turbo-instruct", 
    temperature=0, 
    openai_api_key=openai_api_key
)

# Define a buffer memory
# Masterful memory management! You may need to experiment with different size 
# values when implementing a buffer memory for your use case, balancing 
# context retention with memory constraints.
memory = ConversationBufferMemory(size=4)

# Define the chain for integrating the memory
# with the model
buffer_chain = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

# Invoke the chain with the inputs provided
buffer_chain.predict(input="Write Python code to draw a scatter plot.")
buffer_chain.predict(input="Use the Seaborn library.")

    
# >Entering new ConversationChain chain...
# Prompt after formatting:
# The following is a friendly conversation between a human and an AI. The AI is 
# talkative and provides lots of specific details from its context. If the AI 
# does not know the answer to a question, it truthfully says it does not know.

# Current conversation:

# Human: Write Python code to draw a scatter plot.
# AI:
# Finished chain.

# Entering new ConversationChain chain...
# Prompt after formatting:
# The following is a friendly conversation between a human and an AI. The AI is 
# talkative and provides lots of specific details from its context. If the AI 
# does not know the answer to a question, it truthfully says it does not know.

# Current conversation:
# Human: Write Python code to draw a scatter plot.
# AI: Sure, I can definitely help you with that! To draw a scatter plot in 
# Python, you will need to import the matplotlib library. This can be done by 
# using the command "import matplotlib.pyplot as plt". Then, you can create a 
# scatter plot by using the "scatter()" function and passing in the x and y 
# values as parameters. For example, if you have two lists of data called 
# "x_data" and "y_data", you can create a scatter plot by using the code 
# "plt.scatter(x_data, y_data)". You can also customize your scatter plot by 
# adding labels, titles, and changing the color and size of the points. Is 
# there anything else you would like to know about drawing a scatter plot in Python?
#
# Human: Use the Seaborn library.
# AI:

# Finished chain.
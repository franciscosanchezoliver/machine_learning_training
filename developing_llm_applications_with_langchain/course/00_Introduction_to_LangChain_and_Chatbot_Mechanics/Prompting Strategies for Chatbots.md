#Chatbot #LangChain #LLM #PromptTemplate #LCEL

Let's **use LangChain** to start implementing **prompting strategies for chatbots**. 

Besides OpenAI's **chat models**, thousands of chat-optimized language models **are available in LangChain via the Hugging Face Hub API**.

**To find language models specifically optimized for chat, search the chat models sections of Hugging Face and filter on Question Answering**

![Hugging Face Question Answering Models](Pasted%20image%2020240615062119.png)

Then, take **note** of **the model name so it can be referenced in the code**.

New models are constantly being released on Hugging Face. Many are also fine-tuned on domain-specific datasets, so they are better at capturing the nuance of a particular region, culture, or task, so **searching Hugging Face for the most appropriate model for the use case** will be worthwhile.

**Once we've selected a model, we can begin prompting it by utilizing prompt template.**

## Prompt Templates

**Prompt template** acts as **reusable recipes for generating prompts from user inputs in a flexible and modular way.** 
![[Pasted image 20240831084156.png]]

**A template can include instructions, examples**, and any **additional context that** might **help the model complete the task**. 

**Prompt templates are created using** LangChain's **"*PromptTemplate*" class**. 

```python
# Depending on the version of LangChain the imports may vary
from langchain_core.prompts import PromptTemplate

# Create a template string
# The "question" field is defined for dynamic insertion later in the code
template = "You are an artifical intelligence assistant, answer the question. {question}"

prompt_template = PromptTemplate(
			template=template, 
			input_variables=["question"]
		)

# To show how a variable will be inserted, let's call the method 'invoke'
# and pass a dictionary to map values to input.
result = prompt_template.invoke({
	"question": "What is LangChain?"
})

print(result)
```

Now that we have our prompt template, **let's integrate it into an LLM**. 
We start by choosing an question-answering LLM from Hugging Face.

```python
from langchain_hugginface import HuggingFaceEndpont

# We start by choosing an LLM from Hugging Face, in this case, 
# a falcom model 
llm = HuggingFaceEndpoint(
	repo_id="tiiuae/falcon-7b-instruct",
	huggingfacehub_api_token=hugging_face_api_key 
)
```

**To integrate the a Prompt Template and a Model,** we use Lang Chain Expression Languague (**LCEL**).
This allow us to **create a Pipe (or Chain)**, which in LangChain **is used to connect a series of calls to different components into a sequence**.

```python
llm_chain = prompt_template | llm

# To begin passing user inputs, we call the run() method 
# on the chain passing the input string
question = "What is LangChain?"

# To pass an input to this chain, we call the invoke method again with the
# same dictionary as before
llm_chain.invoke({
	"question": question		
})
```

Note 1: the class *HuggingFaceHub* is deprecated and you should use *HuggingFaceEndpoint* instead
Note 2: The LLMChain is deprecated since version 0.1.17, use "prompt | llm" instead.
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(
    input_variables=["adjective"], 
    template=prompt_template
)
llm = OpenAI()
chain = prompt | llm

result = chain.invoke("your adjective here")
print(result)
```

So using the non deprecated methods:
```python
import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(
    input_variables=["adjective"], 
    template=prompt_template
)

hugging_face_api_key = os.environ["huggingface_apikey"]
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=hugging_face_api_key
)
chain = prompt | llm
output = chain.invoke("black humor")
print(output)
```

## Chat Models

Chat Models have a different prompt template class: ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

# This allow us to specify a series of messages to pass to the
# chat model.
# This is structured as a list of tuple, where each tuple contains
# a role and a message pair.
# We create a prompt_template by passig this list of tuples to the 
# "from_messages" method.
# In this example we can se 3 roles:
# - System
# - User
# - AI
# The system role is used to defined the model behaviour, the human 
# role is used for providing inputs, and the AI role is used for 
# defining outputs which is often used to provide additional examples
# for the model
# 
prompt_template = ChatPromptTemplate.from_messages([
	("system", "You are soto zen master Roshi."),
	("human", "What is the essence of Zen?"),
	("ai", "When you are hungry, eat. When you are tired, sleep."),
	("human", "Respond to the question: {question}")
])


# The ChatOpenAI class is used to access OpenAI's models.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
		model="gpt-4o-mini",
		openai_api_key=openai_apikey
	)

llm_chain = prompt_template | llm

question = "What is the sound of one hand clapping?"

response = llm_chain.invoke({
				"question": question
			})
print(response)
```

[Code Example](01_chain_example.py)

[Code Example 02](02_prompt_template_and_chaining.py)

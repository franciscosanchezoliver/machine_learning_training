# Environment
As the llm environment is pretty huge, I recommend to create an environment
and share it between all the projects. 

I've moved the .venv to ~/shared_llm_env

To reuse it, just activate it by: 
```
source ~/shared_llm_env/bin/activate
```

# Exercise Notes

1. The LangChain ecosystem

3. Build LLM Apps with LangChain

- Released in 2022
- Integration between AI and LLMs.
- Used to build LLM-powered applications.

4. The LangChain ecosystem

- LangSmith -> Deploy Application into production
- LangGraph -> Creating AI Agents

5. LangChain integrations

- Integration with a lot of AI models and Databases (via LangChain)

- Langchain Providers documentation: https://python.langchain.com/docs/integrations/providers/

6. Building LLM apps the LangChain way...

Let's say that we want to create a customer support chatbot that uses LLMs 
to converse with customers. 
Requirements:
- The chatbot needs to be able to provide product information and recommendations
- Responding to customers experiencing issues with placing orders. 
- Ensure that responses by the model should be based on existing support 
  articles, which can be easily tweaked and maintained.

There's a few different components to manage here: 
- An LLM, which may be from a whole host of different providers, 
  both proprietary and open-source;
- A mechanism to help the model decide whether to provide product information 
  or advise on troubleshooting issues,
- A database of customer support articles for the model to use.
- A mechanism for finding and integrating them into the chatbot. 

Throughout the course, we'll use LangChain to create these components and 
connect them together in a modular and intuitive way.

11. Prompting OpenAI models

We'll begin our LangChain journey by defining and prompting LLMs, starting with 
proprietary models from OpenAI. 

The ChatOpenAI class from the langchain_openai partner library can be used to 
define a model to use in LangChain apps. 

This makes a request to the OpenAI API and returns the response back to the 
application. 

OpenAI's API requires an API key, which can also be specified in this class, 
which will incur some cost for its use. Like other DataCamp courses you may 
have taken, you don't have to create an OpenAI account or incur any costs in 
this course - a placeholder API key will be provided for you. 

To prompt this model, we call the .invoke() method on a prompt string. 

The output is pretty long here, but the ChatOpenAI class accepts parameters 
like max_completion_tokens and temperature that you may have 
encountered elsewhere.

https://platform.openai.com/docs/quickstart

12. Prompting Hugging Face models

If we'd rather work with open-source models downloaded into a local directory, 
Hugging Face is an excellent choice for finding an appropriate model. 

The HuggingFacePipeline class and its .from_model_id() can be used to download 
a model for a particular task; here, a text generation model. To pass this 
model a prompt, we again use the .invoke() method. 

Notice that, although we used a completely different model from a different 
model provider, and downloaded it locally instead of making a request to an 
API, we only needed to change one class and its arguments.

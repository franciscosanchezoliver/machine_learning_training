#LLM #RAG #vector_databases #ChromaDB

Now that we've covered document loading and splitting, we'll round-out the RAG workflow with learning about **storing and retrieving** this **information using vector databases**.

**We've loaded documents and split them into chunks using an appropriate chunk_size and chunk_overlap**.

![](Pasted%20image%2020240726075143.png)

All that's left is to store them for retrieval. We'll be using a **vector database to store our documents and make them available for retrieval**.

![](Pasted%20image%2020240727065129.png)

This **requires embedding our text documents to create vectors that capture the semantic meaning of the text**. 

**Then, a user query can be embedded to retrieve the most similar documents from the database and insert them into the model prompt**.

There are many vector databases available in LangChain.

![](Pasted%20image%2020240727070059.png)

When **making the decision on which solution to use**, **consider whether an open source solution is required**, which may be the case **if high customizability is required**.

Also, consider whether the data can be stored on off-premises on third party services (not all cases will permit this).

**The amount of storage and latency of retrieving results is also a key consideration.**

**Sometimes a lightweight in-memory database will be sufficient**, but others will require something more powerful.

Fort this part, we'll be using "ChromaDB" because it is a lightweight and quick to set up.

We'll be storing documents containing guidelines for a company's marketing copy. There are two guidelines: one around brand capitalization, and another how to refer to users.

```
[
	Document(
		page_content="In all marketing copy, TechStack should always be written 
		with the T and S capitalized. Incorrect: techstack, Techstack, etc.",
		metadata={"guideline":"brand-capitalization"}
	),
	Document(
		page_content="Our users should be referred to as techies in both internal
		and external communications.",
		metadata={"guidelines":"referring-to-users"}
	)
]
```

**Now that we've parsed the data, it's time to embed it.**

We'll use an embedding model from OpenAI by instantiating the OpenAIEmbeddings
class.
```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Create a Chroma database from a set of documents
vectorstore = Chroma.from_documents(
				docs, # documents we want to conver to vector
				embedding=embedding_function, # embedding function to use
				# Persist this database to disk for future use
				persist_directory="path/to/directory"
			)

# Finally, to integrate the database with other LangChain components
# we need to convert it into a retriever 
retriever = vectorstore.as_retriever(
				# We want to perform a simlitary search
				seach_type="similarity", 
				# Return the top 2 most similar documents, 
				# for each user query
				search_kwargs={"k": 2}						 
			)

# So the model knows what to do, we'll construct a prompt template,
# which starts with the instruction: to review and fix the copy
# provided, insert the retrieved guidelines and copy to review 
# and an indication that the model should follow with a fixed version
message = """
Review and fix the following TechStack marketing copy with the following guidelines in consideration:

Guidelines:
{guidelines}

Copy:
{copy}

Fixed Copy:
"""

prompt_template = ChatPromptTemplate.from_messages([("human", message)])

# To chain together our retriever, prompt_template, and LLM, we use 
# LCEL in a similar way as before, using pipes to connect the three 
# components. 
# The only difference is that we create a dictionary that assigns the
# retrieved to guidelines, and assigns the copy to review to the 
# RunnablePassthrough function, which acts as a placeholder to insert
# our input when we invoke the chain
from langchain_core.runnables import RunnablePassthrough

rag_chain = ({"guidelines": retriever, "copy": RunnablePassthrough()}
			  | prompt_template
			  | llm)

# Printing the result, we can see the model fixed the two guidelines 
# breaches
response = rag_chain.invoke("Here at techstack, our users are the best in the world!")

print(response.content)
# Output:
# Here at TechStack, our techies are the best in the world!

```






We will talk about loading and preparing external data sources for a chat model, the first step of a process called Retrieval Augmented Generation (RAG).

Pre-trained language models do not have access to private or proprietary data sources. 

We call the process of **integrating these data sources Retrieval Augmented Generation** (or RAG for short).

1. This starts with a user query. The user query is sent to an application created with a framework such as LangChain and transformed the query to a vector representation.
2. This model searches for the most relevant document in a vector database by comparing them to the user's query.
3. It then ranks the most relevant results to the user's query using a chosen distance metric.
4. Finally, the most relevant documents from the vector database are concatenated to the user query and sent to the model.
5. The model generates a response, which is sent back to the end user through the application.

There are 3 primary strep to RAG development in LangChain.
1. Loading the documents into LangChain with document loaders.
2. Next, splitting the documents into chunks. Chunks are units of information that we can index and process individually.
3. The last step is encoding and storing the chunks for retrieval, which could utilize a vector database if that meets the needs of the use case.

We'll discuss all of these steps throughout the next chapter, but for now we'll start with document loaders.

LangChain has more than 160 document loaders. Some loaders are provided by 3rd parties who manage unique document formats. These includes Amazon S3, Microsoft, Google Cloud, Jupyter notebooks, pandas DataFrames, unstructured HTML, YouTube audio transcripts, and more.

We will practice loading the data from three common formats. LangChain has excellent documentation on all of its documents loaders, and you'll find that the implementations are very similar for different formats.

There are many types of PDF loaders in LangChain, there is documentation available for each. For this section, we'll use "PyPDFLoader". This class loads one document for each page, including the PDF metadata.

```python

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("attention_is_all_you_need.pdf")
data = loader.load()
print(data[0])

# Output:
# Document(page_content="Provided proper attribution is provided, Google
#          hereby grants permission to reproduce the tables and figures
#          in this paper solely for use in ...
# )
```

[Reading a PDF example](../06_PDF_document_loaders.py)

When loading CSVs, the syntax is very similar, but instead we use CSVLoader class.

Different document loaders work with various document formats. But in general, document loading follows the same syntax.

For third-party document formats, there are many libraries available.

As an example, we can use Hacker News Loader to get the top Hacker News stories from the URL.

```python
from langchain_community.document_loaders import HNLoader

loader = HNLoader("https://news.ycombinator.com")
data = loader.load()

print(data[0])
# Output:
# Document(page_content="Nrsc5: Receive NRSC-5 digital, radio stations...)

print(data[1]) # Document Metadata
# {'source': 'http://news.ycombinator.com'
#  'title': 'Nrsc5': Receive NRSC-5 digital}
```

[Reading Hacker news with Third party loader](../06_third_party_document_loaders.py)






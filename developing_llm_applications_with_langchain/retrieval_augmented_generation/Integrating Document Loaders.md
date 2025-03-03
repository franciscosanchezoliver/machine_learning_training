#RAG #LLM #LangChain 
#document_loaders #pdf_loader #csv_loader 

In this chapter we'll discuss **Retrieval Augmented Generation or RAG**.

**Pre-trained language models doesn't have access to external data sources**, **their understanding comes** purely **from their training data**. This means that **if we require our model to have knowledge that goes beyond its training data**, which could be company data or knowledge of more recent world events, **we need a way of integrating that data**.

In RAG, **an user query is embedded and used to retrieve the most relevant documents from the database**, then **these documents are added to the model's prompt so that the model has extra context to inform its response**.

![](Pasted%20image%2020240715054605.png)

There are **3 primary steps to RAG development in LangChain**:
1. **Loading the documents into LangChain with document loaders**.
2. **Split the documents into chunks**. **Chunks are units of information that can be index and process individually**.
3. The last step is **encoding and storing the chunks for retrieval**, which **could utilize a vector database** if that meets the needs of the use case.

We'll discuss all of these steps in the next chapter, but for now, we'll start with documents loaders.
## LangChain Document Loaders

**LangChain document loaders are classes designed to load** and configure **documents for integration with AI systems**. **LangChain provides** document loaders **classes for common file types** such as CSV and PDF.
There are also **additional loaders provided by 3rd parties for managing unique documents** (such as .ipynb, .wav, s3 files, audio transcripts...)
In this document, we'll practice loading data from the common formats: PDFs, CSVs, and HTML.
LangChain has excellent documentation on all its document loaders, and there's are lot of overlap in syntax, so explore at your leisure.
## PDF Document Loader

There are a few types of PDF loaders in LangChain, and there is documentation available online for each.
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path/to/file/attention_is_all_you_need.pdf")
data = loader.load()
print(data[0])
```

[Exercise](../06_PDF_document_loaders.py)

## CSV Document Loader

When loading CSVs, the syntax is very similar, but instead we use the CSVLoader class.

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader("fifa_countries_audience.csv")
data = loader.load()
print(data[0])
```
[Exercise](../06_CSV_document_loaders.py)


## HTML Document Loader

Finally, we can load HTML files using UnstructuredHTMLLoader class.

```python
from langchain_community.document_loaders import UnstructuredHTMLLoader

loader = UnstructuredHTMLLoader("white_house_executive_order_nov_2023.html")
data = loader.load()

print(data[0])
```

[Exercise](../15_HTML_document_loaders.py)

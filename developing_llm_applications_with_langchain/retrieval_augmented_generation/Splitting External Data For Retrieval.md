#splitter #llm #LangChain #CharacterTextSplitter
Now that we've loaded documents from different sources, let's learn how to parse the information.
![](Pasted%20image%2020240724080506.png)

**Document splitting, splits the loaded documents into chunks**. 

**Chunking** is particularly useful for **breaking up long documents so that they fit within an LLM's context window**.

Let's examine the introduction from an academic paper, which is saved as PDF.

```
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine learning translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures.
```

**One naive splitting option would be to separate the document line by line**:
Line 1:
```
Recurrent neural networks, long short-term memory and gated recurrent neural networks
```

Line 2
```
in particular, have been firmly established as state of the art approaches in sequence modeling and
```

This would be simple to implement, but because **sentences are often split over multiple lines**, and because those lines are processed separately, **key context might be lost**.

**To counteract lost context during chunk splitting, a chunk overlap is often implemented**.

Chunk 1
```
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly
```

Chunk Overlap (Shared between the other two chunks)
```
established as state of the art approaches in sequence modeling and transduction problems
```

Chunk 2
```
Such as language modeling and machine translation. Numerous efforts have since continued to push the bounderies of recurrent language models and encoder-decoder architectures
```

We've selected two chunks and a chunk overlap. 

Having this extra overlap present in both chunks helps retain context. 

If the model shows signs of loosing context and misunderstanding information when answering from external sources, we may need to increase the chunk overlap.

There isn't one document splitting strategy that works for all situations. We should experiment with multiple methods, and see which one strikes the right balance between retaining context and managing chunk size. 

We will compare two splitting methods: 
- CharacterTextSplitter
- RecursiveCharacterTextSplitter

Optimizing this document splitting is an active area of research, so keep an eye out for new developments.

As an example, let's split this quote by Elbert Hubbard, which contains 103 characters, into chunks.

```python
quote = "One machine can do the work of fifty ordinary humans. \nNo machine can do the work of one extraordinary human." 

len(quote) # 103
```

We'll compare how the two methods perform on this quote with a chunk size of 24 characters and a small chunk overlap of 3.

### CharacterTextSplitter

This method splits based on the separator first, then evaluates chunk_size and chunk_overlap to check if it's satisfied.

```python
from langchain.text_splitter import CharacterTextSplitter

ct_splitter = CharacterTextSplitter(
				separator=".",
				chunk_size=chunk_size,
				chunk_overlap=chunk_overlap
			)

docs = ct_splitter.split_text(quote)
print(docs)

# [
#  "One machine can do the work of fifty ordinary humans", 
#  "No machine can do the work of one extraordinary human"
# ]

# print len of each chunk
print([len(doc) for doc in docs])
# [52, 53]
```

We can see that each of these chunks contains more characters than our specified chunk_size.  

CharacterTextSplitter splits on the separator in an attempt to make chunks smaller than chunk_size, splitting on the separator was unable to return chunks below our chunk size.


Let's take at a more robust splitting method.

### RecursiveCharacterTextSplitter

Takes a list of separators to split on, and it works through the list from left to right, splitting the document using each separator in turn, and seeing if these chunks can be combined while remaining under chunk_size

```python

from langchain.text_splitter import RecursiveCharacterTextSplitter

rc_splitter = RecursiveCharacterTextSplitter(
				separators=["\n\n", "\n", " ", ""],
				chunk_size=chunk_size,
				chunk_overlap=chunk_overlap
				)

docs = rc_splitter.split_text(quote)
print(docs)
# [
#  "One machine can do the",
#  "work of fifty ordinary", 
#  "humans.",
#  "No machine can do the",
#  "work of one", 
#  "extraordinary human."
# ]
```

Notice how the length of each chunk varies. 

The class splits by paragraphs first, and found that the chunk size was too big, likewise for sentences. I got to the third separator: splitting words using the space separator, and found that words can be combined into chunks while remaining the chunk_size character limit.

However, some of these chunks are too small to contain meaningful context, but this recursive implementation may work better on larger documents.


We can also use split other file formats, like HTML (recall that we can load HTML using UnstructuredHTMLLoader)

```python

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = UnstructuredHTMLLoader("white_house_executive_order_nov_2023.html")

data = loader.load()

rc_splitter = RecursiveCharacterTextSplitter(
				chunk_size=chunk_size,
				chunk_overlap=chunk_overlap,
				separators=["."])

# To split documents, we use the "split_documents" method instead of
# the "split_text" to perform the split.
docs = rc_splitter.split_documents(data)
print(docs[0])
```




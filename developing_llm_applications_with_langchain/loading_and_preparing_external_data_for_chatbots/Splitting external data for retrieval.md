#LLM #LangChain 

**Now that we've loaded documents from different sources**, let's learn how to **properly parse the information**.

**Document splitting is when we split the loaded document into smaller parts**, which are also called **called "chunks"**.

**Chunking** is particularly **useful for breaking up long documents so that they fit within an LLM's context window**.

Let's examine the introduction from a famous paper called "Attention is All you Need" which is saved as a PDF.

```
Recurrent neural networks, long short-term memory [12] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [29, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [31, 21, 13].
```

One **naive splitting method** would be to **separate the document into lines** as they appear in the paper. This **would** be **simple to implement but** could be **problematic**. 

**Key context required for understanding one line is often found in a different line**, and these lines would be processed separately, so **we need another strategy**. 

**There isn't one strategy that works for all situations when it comes to splitting documents**.

It is often the case of **experimenting with multiple methods, and seeing which one strikes the right balance between retaining sufficient context and managing chunk size**.

We will compare **two document splitting** methods from LangChain:
- **CharacterTextSplitting** **splits text based on a specific separator**, looking at individual characters.
- **RecursiveCharacterTextSplitter** **attempts to split by several characters recursively until the chunks fall within the specified chunk size**.

**There are many other methods that use natural language processing to infer meaning and split appropriately**. Optimizing this is an active area of research.

**When we split the document into chunks, a chunk overlap is needed to ensure any context is properly conveyed across chunks**.

To understand the concept fully, let's go back to the Attention is All You Need introduction.

Imagine that we have the following two overlapping chunks:

```
--- Chunk 1 ---
Recurrent neural networks, long short-term memory, and gated recurrent neural networks in particular, have been firmly 

--- Overlapping Chunk ---
established as state of the art approaches in sequence modeling and transductions problems

--- Chunk 2 ---
such as language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures
```

Having t**his extra overlap in each chunk helps information retain context**.

**If a model shows signs of losing context and misunderstanding information when answering from external sources, we may need need to increase the chunk overlap.** 

As an example, let's split this modernized quote by Elbert Hubbard into chunks, which contains 103 characters.

```python
quote = "One machine can do the work of fifty ordinary humans. No machine can do the work of one extraordinary human."
len(quote) # 103
```

We'll compare how the two methods perform on this quote with a chunk size of 24 and small chunk overlap of three.

Let's start with **CharacterTextSplitter**. **This method splits based on the separator first**, **then evaluates the chunk size and chunk the overlap.**

```python
from langchain.text_splitter import CharacterTextSplitter

chunk_size = 24
chunk_overlap = 3

ct_splitter = CharacterTextSpliter(
				separator='.',
				chunk_size=chunk_size,
				chunk_overlap=chunk_overlap
			  )

docs = ct_splitter.split_text(quote)
print(docs)

# [
#  'One machine can do the work of fifty ordinary humans.',
#  'No machine can do the work of one extraordinary human.'
# ]
```

[Example of Character Text Splitter](./../07_character_text_splitter.py)
[Example 2 of Character Text Splitter](./../07_character_text_splitter_ex_2.py)

Comparing to the **RecursiveCharacterSpliter**, let's split using the same chunk size and overlap values.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter


quote = "One machine can do the work of fifty ordinary humans. No machine can do the work of one extraordinary human."
len(quote) # 103

chunk_size = 24
chunk_overlap = 3

rc_splitter = RecursiveCharacterTextSplitter(
				chunk_size=chunk_size,
				chunk_overlap=chunk_overlap
			  )

docs = rc_splitter.split_text(quote)
print(docs)
# [
#  'One machine can do the', 
#  'work of fifty ordinary',
#  'humans. No machine can do',
#  'do the work of one',
#  'extraordinary human.'
# ]
```

[Example of Recursive Character Text Splitter](./../07_recursive_character_splitter.py)
[Example of Recursive Character Text Splitter](./../07_recursive_text_splitter_ex2.py)


**Notice how the length of each chunk varies.** **Whether this is good or bad for your data depends on the use case and document**. The chunks for this particular case are too small to contain appropriate context, but the implementation may work better on a larger document.

We **can also use the splitter syntax with other formats, such as PDFs and HTML.** As we learned earlier in the chapter, many document formats have their own document loader classes in LangChain.

```python

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = UnstructuredHTMLLoader("white_house_executive_order_nov_2023.html")
data = loader.load()

# We can then use the splitter to split the document
recursive_character_splitter = RecursiveCharacterSplitter(
								chunk_size=chunk_size,
								chunk_overlap=chunk_overlap,
								separators=['.']
								)

# For HTML and other non-string formats, we use the "split_docuemnts"
# method instead of "split_text" to perform the split.
docs = recursive_character_splitter.split_documents(data)
print(docs[0])

# Document(page_content='To search this site, enter a search term [...])
```

[Example of Reading HTML and splitting with Recursive Character Text Splitter](./../07_HTML_document_splitter.py)
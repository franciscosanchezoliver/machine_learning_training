"""
Splitting by character
----------------------
A key process in implementing Retrieval Augmented Generation (RAG) is splitting 
documents into chunks for storage in a vector database.
There are several splitting strategies available in LangChain, some with more 
complex routines than others. In this exercise, you'll implement a character 
text splitter, which splits documents based on characters and measures the 
chunk length by the number of characters.
Remember that there is no ideal splitting strategy, you may need to experiment 
with a few to find the right one for your use case.

Instructions
- Import the appropriate LangChain class for splitting a document by character.
- Define a character splitter that splits on "\n" with a chunk_size of 24 and chunk_overlap of 10.
- Split quote, and print the chunks and chunk lengths.
"""

from langchain.text_splitter import CharacterTextSplitter

quote = "Words are flowing out like endless rain into a paper cup,\nthey slither while they pass,\nthey slip away across the universe."

chunk_size = 24
chunk_overlap = 10

# Create an instance of the splitter class
splitter = CharacterTextSplitter(
    separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Split the string into chunks and print the chunks
docs = splitter.split_text(quote)

print(docs)
print([len(doc) for doc in docs])


print(
    f"""
Original Text
-------------
{quote}

Split Text (Using CharacterTextSplitter)
----------------------------------------"""
)
for index, doc in enumerate(docs):
    print(f"Chunk {index + 1}")
    print(f"    Size: {len(doc)}")
    print(f"    Content: {doc}\n")

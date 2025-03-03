"""
Recursively splitting by character
----------------------------------
Many developers are using a recursive character splitter to split documents 
based on a specific list of characters. These characters are paragraphs, 
newlines, spaces, and empty strings, by default: ["\n\n", "\n", " ", ""].

Effectively, the splitter tries to split by paragraphs, checks to see if the 
chunk_size and chunk_overlap values are met, and if not, splits by sentences, 
then words, and individual characters.

Often, you'll need to experiment with different chunk_size and chunk_overlap 
values to find the ones that work well for your documents.

Instructions:
- Import the appropriate LangChain class for splitting a document 
  recursively by character.
- Define a recursive character splitter to split on the characters "\n", " ", 
  and "" (in that order) with a chunk_size of 24 and chunk_overlap of 10.
- Split quote
"""

# Import the recursive character splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

quote = 'Words are flowing out like endless rain into a paper cup,\nthey slither while they pass,\nthey slip away across the universe.'
chunk_size = 24
chunk_overlap = 10

# Create an instance of the splitter class
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)

# Split the document and print the chunks
docs = splitter.split_text(quote)

print(f"""
Text to Split 
-------------
{quote}      

Chunks obtained using RecursiveCharacterTextSplitter
----------------------------------------------------
{docs}

Size of each Chunk
------------------
{[len(doc) for doc in docs]}
""")


# Output:
# [
#     'Words are flowing out', 
#     'out like endless rain', 
#     'rain into a paper cup,', 
#     'they slither while they', 
#     'they pass,', 
#     'they slip away across', 
#     'across the universe.'
# ]
# [21, 21, 22, 23, 10, 21, 20]

# RecursiveCharacterTextSplitter was able to keep the chunks below chunk_size, 
# albeit with a few chunks containing little meaning. Take a moment to experiment 
# with different chunk_size and chunk_overlap values, running the code each 
# time and interpreting the results. 
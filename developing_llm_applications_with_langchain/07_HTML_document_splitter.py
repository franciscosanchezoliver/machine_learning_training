
"""
Splitting HTML
--------------
In this exercise, you'll split an HTML containing an executive order on AI 
created by the US White House in October 2023. To retain as much context as 
possible in the chunks, you'll split using larger chunk_size and chunk_overlap 
values.
All of the LangChain classes necessary for completing this exercise have been pre-loaded for you.

Instructions:
- Create a document loader for white_house_executive_order_nov_2023.html, and 
  load it into memory.
- Set a chunk_size of 300 and a chunk_overlap of 100.
- Define the splitter, splitting on the '.' character, and use it to split data 
  and print the chunks.
"""
import os
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


html_file_path = os.path.join(os.getcwd(), "data", "white_house_executive_order_nov_2023.html")
loader = UnstructuredHTMLLoader(html_file_path)
data = loader.load()

# Define variables
chunk_size = 300
chunk_overlap = 100

# Split the HTML
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators="."
)

docs = splitter.split_documents(data)
print(docs)

print(f"""
Load HTML with UnstructuredHTMLLoader
-------------------------------------
{data }
""")
print(f"""
Split Documents with RecursiveCharacterTextSplitter
--------------------------------------------------
{docs}
""")
"""
HTML document loaders
---------------------
It's possible to load documents from many different formats, including complex 
formats like HTML.

In this exercise, you'll load an HTML file containing a White House 
executive order.

Instructions:
- Use the UnstructuredHTMLLoader class to load the white_house_executive_order_nov_2023.html file in the current directory.
- Load the documents into memory.
- Print the first document.
- Print the first document's metadata.
"""

import os
from langchain_community.document_loaders import UnstructuredHTMLLoader

html_file_path = os.path.join(os.getcwd(), "data", "white_house_executive_order_nov_2023.html")
loader = UnstructuredHTMLLoader(html_file_path )
data = loader.load()

print(f"""
First Page Metadadata
---------------------
{data[0].metadata}

First Page
----------
{data[0]}
""")

"""
CSV document loaders
--------------------
Comma-separated value (CSV) files are an extremely common file format, 
particularly in data-related fields. Fortunately, LangChain provides different 
document loaders for different formats, keeping almost all of the syntax the 
same!

In this exercise, you'll use a document loader to load a CSV file containing 
data on FIFA World Cup international viewership. 

Instructions
- Import the appropriate class for loading CSV documents in LangChain.
- Create a document loader for the 'fifa_countries_audience.csv' document, 
  which is available in the current directory.
- Load the documents into memory to view the contents of the first document.
"""

import os
from langchain_community.document_loaders.csv_loader import CSVLoader


csv_file_path = os.path.join(os.getcwd(), "data", "fifa_countries_audience.csv")
loader = CSVLoader(file_path=csv_file_path)

# Load the document
# CSVLoader creates a document for each row in the CSV file, which could be 
# used in an LLM application to summarize data and generate reports.
data = loader.load()
print(data[0])


print(f"""
-----------------------
CSV Content (firt page)
-----------------------
{data[0]}

-------------
File Metadata
-------------
{data[0].metadata}
""")


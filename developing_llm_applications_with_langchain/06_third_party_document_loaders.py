"""
Third-party document loaders
----------------------------

It's possible to load documents from many different formats, including formats 
that may be proprietary or dictated by an outside organization.

In this exercise, you'll utilize the Hacker News document loader, HNLoader, 
which is used to access news stories from their website.

Instructions
- Use the HNLoader class to create a document loader for the top Hacker News 
stories from "https://news.ycombinator.com".
- Load the documents into memory.
- Print the first document.
- Print the first document's metadata.
"""

from langchain_community.document_loaders import HNLoader

# Create a document loader for the top Hacker News stories
hacker_news_web_page = "https://news.ycombinator.com"
loader = HNLoader(web_path=hacker_news_web_page)

# Load the document
data = loader.load()

# Print the first document
print(data[0])

# Print the first document's metadata
print(data[0].metadata)


# CONTENT
# page_content='Inside a $1 radar motion sensor (10maurycy10.github.io)' 
# metadata={'source': 'https://news.ycombinator.com', 'title': 'Inside a $1 
# radar motion sensor (10maurycy10.github.io)', 
# 'link': 'https://10maurycy10.github.io/projects/motion_sensor_hacking/', 'ranking': '1.'

# METADATA
# {'source': 'https://news.ycombinator.com', 'title': 'Inside a $1 radar motion 
# sensor (10maurycy10.github.io)', 
# 'link': 'https://10maurycy10.github.io/projects/motion_sensor_hacking/', 
# 'ranking': '1.'}


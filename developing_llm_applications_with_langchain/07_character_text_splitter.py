from langchain.text_splitter import CharacterTextSplitter

chunk_size = 24
chunk_overlap = 3

quote = "One machine can do the work of fifty ordinary humans. No machine can do the work of one extraordinary human."

character_splitter = CharacterTextSplitter(
    separator=".", chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

docs = character_splitter.split_text(quote)

print(
    f"""
Original Text
-------------
{quote}

Split Text (Using CharacterTextSplitter)
----------------------------------------
{docs}
"""
)

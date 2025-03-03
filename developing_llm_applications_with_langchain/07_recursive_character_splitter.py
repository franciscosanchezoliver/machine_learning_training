from langchain.text_splitter import RecursiveCharacterTextSplitter

quote = "One machine can do the work of fifty ordinary humans. No machine can do the work of one extraordinary human."

chunk_size = 24
chunk_overlap = 3

recursive_character_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

splits = recursive_character_splitter.split_text(quote)

print(
    f"""
Original Text
-------------
{quote}

Split Text (Using RecursiveCharacterTextSplitter)
Chunk Size: {chunk_size}
Chunk Overlap: {chunk_overlap}
-------------------------------------------------
"""
)

for index, split in enumerate(splits):
    print(f"Chunk {index + 1}")
    print(f"    Size: {len(split)}")
    print(f"    Content: {split}\n")

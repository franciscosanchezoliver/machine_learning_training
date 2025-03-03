import os
from langchain_openai import OpenAI

openai_apikey = os.environ["openai_apikey"]

llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_apikey
)

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.invoke(question)
print(output)


# One of the simplest ways to maintain the quality and appearance of your shoes is to take care of them regularly. Here are some tips to help you do just that:
# 1. Clean them regularly: Dirt and debris can easily build up on your shoes, especially if you wear them often. Use a soft brush or cloth to remove any dirt or dust from the surface of your shoes. For tougher stains, use a mild soap and water solution or a specialized shoe cleaner.
# 2. Protect them from water and moisture: Wet shoes can not only cause discomfort but also damage the material of your shoes. If you get caught in the rain, make sure to dry your shoes thoroughly before wearing them again. You can also use a waterproof spray to protect your shoes from water and moisture.
# ...
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser

# As we are going to use a OpenAI LLM, we need to set the API key
openai_apikey = os.environ["openai_apikey"]

destination_prompt = PromptTemplate(
    input_variables=["destination"],
    template="""
    I am planning a trip to {destination}. Can you suggest some activities
    to do there?
    """,
)

activities_prompt = PromptTemplate(
    input_variables=["activities"],
    template="""
    I only have one day, so can you create an itinerary from your 
    top three activities: {activities}
    """,
)

llm = ChatOpenAI(openai_api_key=openai_apikey)

seq_chain = (
    {"activities": destination_prompt | llm | StrOutputParser()}
    | activities_prompt
    | llm
    | StrOutputParser()
)

print(seq_chain.invoke({"destination": "Almeria"}))

# Output:
#
# Here's a suggested itinerary for your one-day trip to Almeria:
# Morning:
# - Start your day by visiting the Alcazaba of Almeria. Spend some time
#   exploring the fortress and taking in the views of the city and sea.
# - After visiting the Alcazaba, head to the Cabo de Gata-Nijar Natural Park.
#   Take a hike along one of the trails or relax on one of the beautiful
#   beaches in the area.
#
# Afternoon:
# - Enjoy a relaxing afternoon on one of the beaches in Almeria, such as Playa
#   de los Muertos or Playa de Monsul. Take a swim in the crystal-clear
#   waters or simply soak up the sun.
# - In the afternoon, take a tour of the Tabernas Desert and visit the Mini
#   Hollywood film set. Explore the unique landscape of Europe's only desert
#   and learn about the history of the area.
#
# Evening:
# - End your day by visiting the Cathedral of Almeria. Take a stroll around
#   the cathedral and admire the beautiful Gothic architecture.
# - Finally, explore the historic center of Almeria. Wander through the narrow
#   streets, visit the charming squares, and enjoy a tapas tour to sample some
#   of the delicious local cuisine.
#
# I hope this itinerary helps you make the most of your day in Almeria!
# Enjoy your trip!

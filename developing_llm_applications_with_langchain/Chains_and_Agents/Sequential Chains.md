#LLM #sequential_chains

**Some problems can only be solved sequentially**. Consider a chatbot used to **create a travel itinerary**.

**We need to tell the chatbot** the travel **destination**, **receive suggestions** on what to see in our trip **and tell the model what activities to select to compile the itinerary**. **This is a sequential problem** a**s require more than one user input, one to specify the destination and another to select the activities**.

In sequential chain **the output from one chain becomes the input to another**. **We'll create two prompt template**: **one to generate suggestions** for activities from the input destination, **and another to create an itinerary** for one day of activities for the models top 3 suggestions.

```python
destination_prompt = PromtTemplate(
	input_variables=["destination"],
	template="I am planning a trip to {destination}. Can you suggest some activities to do there?"
)

activities_prompt = PromptTemplate(
	input_variables=["activities"],
	template="I only have one day, so can you create an itinerary from your top three activities: {activities}."
)

# Define our model and begin with the sequential chain
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Defining a dictionary that passes our destionation prompt template
# to the LLM and parses the output to a string. This gets assign to the
# activities key which is important as this is the input variable 
# to the second prompt template then into the llm and again parse it to
# a string.
seq_chain = ({"activities": destination_prompt | llm | StrOutputParser()}
	| activities_prompt
	| llm
	| StrOutputParser())


# to summarize: the destination prompt is passed the llm to generate the
# activities suggestions and the output is parsed to a string and assign 
# to activities, this is passed to the second activities prompt which is
# passed to the llm to generate the itinerary which is parse as a string

# Let's invoke the chain passing Rome as a destination
print(seq_chain.invoke({"destination": "Rome"}))

# Output: 
# - Morning: 
# 1. Start your day early with a visit to the Colosseum. Take a guided
#    tour to learn about its history and significance
# 2. After exploring the Colosseum, head to the Roman Forum and Palatine Hill
#    to see more of the ancient Rome's ruins.
#
# - Lunch:
# 3. Enjoy a delecious Italian lunch at a local restaurant near the historic
#    center.
# 
# - Afternoon:
# 4. Visit the Vatican City and explore St. Peter's Basilica, the Vatican
#    Museums, and the Sistine Chapel.
# 5. Take some time to wander through the charming streets of Rome, stopping
#    at landmarks like the Pantheon, Trevi Fountain, and Piazza Navona
#
# - Evening:
# 6. Relax in one of Rome's beautiful parks, such as Villa Borghese or the
#    Orange Garden, for a peaceful escape from the bustling city.
# 7. End your day with a leisurely dinner at a local restaurant, indulging 
#    in more Italian cuisine and maybe some gelato.

```

The model consider that we only have one day to explore and would give us the top suggestions.

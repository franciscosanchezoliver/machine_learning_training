"""
ReAct agents
------------
Time to have a go at creating your own ReAct agent! Recall that ReAct stands 
for Reason and Act, which describes how they make decisions. In this exercise, 
you'll load the built-in wikipedia tool to integrate external data from 
Wikipedia with your LLM.

Note: The wikipedia tool requires the wikipedia Python library to be installed 
as a dependency, which has been done for you in this case.

Instructions:
- Load the "wikipedia" tool.
- Define a ReAct agent, passing it the model and tools to use.
- Run the agent on the input provided and print the content from the final 
  message in response
"""
import os
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools
from langgraph.prebuilt import create_react_agent

openai_apikey = os.environ["openai_apikey"]

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0, 
    openai_api_key=openai_apikey 
)

# We are going to use wikipedia as a source of information.
tools = load_tools(["wikipedia"])

# Create an agent that uses as LLM OpenAI but as tool the wikipedia data.
agent = create_react_agent(
            model=llm, 
            tools=tools)


# Invoke the agent
response = agent.invoke(
    {
        "messages": [("human", "Summarize key facts about London, England.")]
    }, 
)
print(response)
print(response["messages"][-1].content)

# {
    # 'messages': [
        # HumanMessage(
            # content='Summarize key facts about London, England.', 
            # id='18d9a0cf-ce61-4c8e-8050-a500f8182519'
        # ), 
        # AIMessage(
            # content='', 
            # additional_kwargs={'tool_calls': [{'id': 'call_S92yQ11IMGIajCzCUrAbCW2H', 'function': {'arguments': '{"query":"London, England"}', 'name': 'wikipedia'}, 'type': 'function'}]}, 
            # response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 93, 'total_tokens': 108}, 
                            #    'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 
                            #    'finish_reason': 'tool_calls', 
                            #    'logprobs': None}, 
            # id='run-5f14b309-8412-4943-aee6-d754d56167a0-0', 
            # tool_calls=[{'name': 'wikipedia', 'args': {'query': 'London, England'}, 'id': 'call_S92yQ11IMGIajCzCUrAbCW2H'}], 
            # usage_metadata={'input_tokens': 93, 'output_tokens': 15, 'total_tokens': 108}), 
            # ToolMessage(content="Page: London\nSummary: London ( LUN-dən) is the capital and largest city of both England and the United Kingdom, with a population of around 8.8 million. The wider metropolitan area is the largest in Western Europe, with a population of 14.9 million. London stands on the River Thames in southeast England, at the head of a 50-mile (80 km) estuary down to the North Sea, and has been a major settlement for nearly 2,000 years. Its ancient core and financial centre, the City of London, was founded by the Romans as Londinium and has retained its medieval boundaries. The City of Westminster, to the west of the City of London, has been the centuries-long host of the national government and parliament. London grew rapidly in the 19th century, becoming the world's largest city at the time as it expanded and absorbed the surrounding county of Middlesex alongside parts of Surrey and Kent. In 1965, it was combined with parts of Essex and Hertfordshire to create the administrative area of Greater London, which is governed by 33 local authorities and the Greater London Authority. \nAs one of the world's major global cities, London exerts a strong influence on world art, entertainment, fashion, commerce, finance, education, healthcare, media, science, technology, tourism, transport, and communications. Despite a post-Brexit exodus of stock listings from the London Stock Exchange, London remains Europe's most economically powerful city and one of the world's major financial centres. It hosts Europe's largest concentration of higher education institutions, some of which are the highest-ranked academic institutions in the world: Imperial College London in natural and applied sciences, the London School of Economics in social sciences, and the comprehensive University College London. It is the most visited city in Europe and has the world's busiest city airport system. The London Underground is the world's oldest rapid transit system.  \nLondon's diverse cultures encompass over 300 languages. The 2023 population of Greater London of just under 10 million made it Europe's third-most populous city, accounting for 13.4% of the United Kingdom's population and over 16% of England's population. The Greater London Built-up Area is the fourth-most populous in Europe, with about 9.8 million inhabitants as of 2011. The London metropolitan area is the third-most populous in Europe, with about 14 million inhabitants as of 2016,  making London a megacity.\nFour World Heritage Sites are located in London: Kew Gardens; the Tower of London; the site featuring the Palace of Westminster, Church of St Margaret, and Westminster Abbey; and the historic settlement in Greenwich where the Royal Observatory defines the prime meridian (0° longitude) and Greenwich Mean Time. Other landmarks include Buckingham Palace, the London Eye, Piccadilly Circus, St Paul's Cathedral, Tower Bridge, and Trafalgar Square. The city has the most museums, art galleries, libraries, and cultural venues in the UK, including the British Museum, National Gallery, Natural History Museum, Tate Modern, British Library, and numerous West End theatres. Important sporting events held in London include the FA Cup Final, the Wimbledon Tennis Championships, and the London Marathon. It became the first city to host three Summer Olympic Games upon hosting the 2012 Summer Olympics.\n\nPage: Greater London\nSummary: Greater London is the administrative area of London, England, which is coterminous with the London region. It contains 33 local government districts: the 32 London boroughs, which form a ceremonial county also called Greater London, and the City of London. The Greater London Authority is responsible for strategic local government across the region, and regular local government is the responsibility of the borough councils and the City of London Corporation. Greater London is bordered by the ceremonial counties of Hertfordshire to the north, Essex to the north-east, Kent to the south-east, Surrey to the ", name='wikipedia', id='81e39438-5155-483a-b760-8b9160fa5db8', tool_call_id='call_S92yQ11IMGIajCzCUrAbCW2H'), AIMessage(content="London is the capital and largest city of both England and the United Kingdom, with a population of around 8.8 million. It is situated on the River Thames in southeast England and has been a major settlement for nearly 2,000 years. London is a global city that influences art, entertainment, fashion, commerce, finance, education, healthcare, media, science, technology, tourism, transport, and communications. It remains Europe's most economically powerful city and hosts Europe's largest concentration of higher education institutions. London is known for its diverse cultures, with over 300 languages spoken, and is home to numerous landmarks, museums, art galleries, and cultural venues. The city has four World Heritage Sites and has hosted major sporting events like the Summer Olympics. Greater London, the administrative area of London, encompasses 33 local government districts and is bordered by several ceremonial counties.", 
            # response_metadata={'token_usage': {'completion_tokens': 176, 'prompt_tokens': 928, 'total_tokens': 1104}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-7ac6ba93-5f6d-49eb-b665-a8d145b6a942-0', usage_metadata={'input_tokens': 928, 'output_tokens': 176, 'total_tokens': 1104})
        # ]
# }

# Here are some key facts about London, England:
# 
# 1. London is the capital and largest city of England and the United Kingdom.
# 2. It is located on the River Thames in southeastern England.
# 3. London is a leading global city in the arts, commerce, education, entertainment, fashion, finance, healthcare, media, professional services, research and development, tourism, and transportation.
# 4. The city is known for its iconic landmarks such as the Tower of London, Buckingham Palace, the London Eye, and Big Ben.
# 5. London is a diverse and multicultural city, with a rich history dating back to Roman times.
# 6. The city is home to numerous museums, galleries, theaters, and cultural institutions.
# 7. London is a major financial center and is home to the London Stock Exchange.
# 8. The city hosted the Summer Olympics in 1908, 1948, and 2012.
# 9. London has a temperate maritime climate with mild winters and cool summers.
# 10. The city has a well-developed public transportation system, including the famous London Underground (the Tube).
# 
# These are just a few highlights about London, a vibrant and historic city with a lot to offer.






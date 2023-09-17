import os
# Setup OpenAI Agent
import openai
from llama_index.agent import OpenAIAgent
from llama_hub.tools.google_calendar.base import GoogleCalendarToolSpec
    

# Retrieve the API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

def calendar_agent(input_prompt):
    # Import and initialize our tool spec
    gcal_tools = GoogleCalendarToolSpec().to_tool_list()
    try:
        agent = OpenAIAgent.from_tools(gcal_tools, verbose=True)
        response = agent.chat(input_prompt).response
    except: 
        response = "Calendar agent failed to process your input. Please try again with a different input string. try setting the number_of_results = 10 or less."
    print(f'Caledar agent response: {response}')
    return response if response else "process completed with code 0"
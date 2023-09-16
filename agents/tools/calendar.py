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

    agent = OpenAIAgent.from_tools(gcal_tools, verbose=False)
    return agent.chat(input_prompt).response
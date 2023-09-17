import dotenv 
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
import re
import os 
import langchain
from langchain.agents import initialize_agent, AgentType
from googleapiclient.discovery import build

from langchain.agents.agent_toolkits import GmailToolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from abc import ABC, abstractmethod
from datetime import date
from llama_hub.tools.google_calendar.base import GoogleCalendarToolSpec
from llama_index.agent import OpenAIAgent
import json
import refresh_google_token
from .baseAssistantAgent import BaseAssistantAgent, CustomPromptTemplate, CustomOutputParser
from .tools.calendar_tools import calendar_agent
# Load the .env file
load_dotenv()

from utils.my_gmail_toolkit import CustomGmailToolkit

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

import logging

logging.basicConfig(level=logging.DEBUG, filename='generalAgent.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')



# Retrieve the API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
#print(openai_api_key)

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in the .env file.")


class GeneralAssistanAgent(BaseAssistantAgent):
    def __init__(self):
        self.set_gmail_tools()
        self.extend_tools()
        super().__init__()
      
    def handle_token(self, token_type:str):
        # Check the input argument and determine the source file
        if token_type == "calendar":
            source_file = 'token_calendar.json'
        elif token_type == "email":
            source_file = 'token_mail.json'
        else:
            print("Invalid token type. Please choose 'calendar' or 'email'.")
            return

        # Read the source file and overwrite the token.json
        refresh_google_token.refresh_google_token(source_file)
        with open(source_file, 'r') as src:
            token_data = json.load(src)
            
        with open('token.json', 'w') as dest:
            json.dump(token_data, dest)

        print(f"token.json refreshed and overwritten with {token_type} token.")
      

    def set_gmail_tools(self):
        #self.handle_token('email')
        credentials = get_gmail_credentials(
            token_file="token_mail.json",
            scopes=["https://mail.google.com/"], #"https://www.googleapis.com/auth/calendar"],
            client_secrets_file="credentials.json",
        )
        api_resource = build_resource_service(credentials=credentials)
        toolkit = GmailToolkit(api_resource=api_resource)
        #toolkit = GmailToolkit()
        self.gmail_tools = toolkit.get_tools()

    def extend_tools(self):
        self.extended_tools = {'Calendar':  Tool(
            name='Calendar',
            func= calendar_agent,
            description='useful for when you need to ead and create new events on the human calendar. The input for this tool is a single string containing the natural language inscruction',
          
            ),
        }
    '''
    def  calendar_tools(self, input_prompt: str) -> str:
        self.handle_token('calendar')

        print(f'[INFO] Calendar tools received: {input_prompt}')
        tool_spec = GoogleCalendarToolSpec()
        llama_index_agent = OpenAIAgent.from_tools(tool_spec.to_tool_list(),  verbose=False)
        return llama_index_agent.chat(input_prompt).response
    '''
    def set_tools(self):
        other_tools = [self._tools['Today Date'], self.extended_tools['Calendar']] 
        return self.gmail_tools + other_tools 
        #return self.gmail_tools
    def set_memory(self):
        memory=ConversationBufferWindowMemory(k=5)
        return memory
    def set_prompt_template(self):
        return ''
    def run_agent(self, input_prompt: str) -> str:
        llm = self.llm #OpenAI(temperature=0)
        agent = initialize_agent(
        tools= self.tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        #return_intermediate_steps=True
        memory = self.memory,
        verbose=True 
        )
        logging.debug(f'agent is {agent}')
        logging.debug(f'agent template {agent.agent.llm_chain.prompt}')
        return agent.run(input_prompt) 

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
from .baseAssistantAgent import BaseAssistantAgent, CustomPromptTemplate, CustomOutputParser
# Load the .env file
load_dotenv()

from utils.my_gmail_toolkit import CustomGmailToolkit

os.environ["ZAPIER_NLA_API_KEY"] = os.environ.get("ZAPIER_NLA_API_KEY", "")

import logging

logging.basicConfig(level=logging.DEBUG, filename='agents.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')



# Retrieve the API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in the .env file.")


class SkillfulAssistantAgent(BaseAssistantAgent):
    def __init__(self):
       
        self.set_gmail_tools()
        super().__init__()
      

    def set_gmail_tools(self):
        toolkit = GmailToolkit()
        self.gmail_tools = toolkit.get_tools()

    def set_tools(self):
        other_tools = [self._tools['Today Date']] 
        return self.gmail_tools + other_tools 
        #return self.gmail_tools
    def set_memory(self):
        memory=ConversationBufferWindowMemory(k=2)
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

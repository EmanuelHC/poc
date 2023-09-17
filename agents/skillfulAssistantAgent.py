import dotenv 
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, ZeroShotAgent, create_csv_agent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain,  PromptTemplate
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
#from chains.bagi import BabyAGI
# Load the .env file
load_dotenv()

from utils.my_gmail_toolkit import CustomGmailToolkit

import discord
from discord.ext import commands
import requests

CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")
TOKEN =  os.environ.get("DISCORD_TOKEN", "")
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_ID", "")
BASE_URL = "https://discord.com/api/v10"

#print(f'channel id{CHANNEL_ID} ')
#print(f'token:{TOKEN}')
#print(f'webhook url: {WEBHOOK_URL} ')

HEADERS = {
    "Authorization": f"Bot {TOKEN}",
    "Content-Type": "application/json",
}


os.environ["ZAPIER_NLA_API_KEY"] = os.environ.get("ZAPIER_NLA_API_KEY", "")

import logging

logging.basicConfig(level=logging.DEBUG, filename='agents.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')



# Retrieve the API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
#print(openai_api_key)

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in the .env file.")


class SkillfulAssistantAgent(BaseAssistantAgent):
    def __init__(self):
        self.set_gmail_tools()
        self.extend_tools()
        super().__init__()
        self.initialize_agent()
        
      

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
        todo_prompt = PromptTemplate.from_template("You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}")
        todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
        self.extend_tools =  {'TODO': Tool(
            name = "TODO",
            func=todo_chain.run,
            description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"
            ),
            'CSV': Tool(
                    name = "CSV",
                    func=self.csv_agent_tool,
                    description="First option to answer questions about csv files. Input: a question about the csv file. Output: an answer to the question. Please be very clear what the question is. Dont worry about the csv file, it is already uploaded into the tool"
            ),
            'Send Discord Message':Tool(
                name="Send Discord Message", 
                func=self._send_discord_message, 
                description="Useful when you need to send a discord message. Input: a message to send."
                ),
            "Read Discord Messages":Tool(
                name="Read Discord Messages",
                func=self._read_discord_messages,
                description="Useful when you need to read the last n messages from a discord channel. Input: Optional the number of messages to read. Default is 1. Output: the last n messages from the channel."
            )

        }
    
    def csv_agent_tool(self, input_prompt: str) -> str:

        agent = create_csv_agent(OpenAI(temperature=0), 
                                'cleaned_file.csv', 
                                verbose=True)
        return agent.run(input_prompt)
    
    def discord_agent(self, input_prompt: str) -> str:
        discord_tools = [
                        ]

        
    

    def _send_discord_message(self, input_prompt: str) -> str:
        """Send a message to the Discord channel using a webhook."""
        data = {"content": input_prompt}
        response = requests.post(WEBHOOK_URL, json=data)
        return response.status_code == 204  # 204 means the message was sent successfully
        
    def _read_discord_messages(self, n: int =10) -> str:
        n = 10#int(n)
        response = requests.get(f"{BASE_URL}/channels/{CHANNEL_ID}/messages?limit={n}", headers=HEADERS)
        print(f'{response} is response')
        logging.debug(f'{response} is response')
        messages = response.json()

        detailed_messages = []
        for message in messages:
            message_content = message['content']
            if not message_content:  # If the message content is empty
                if message['attachments']:
                    message_content = "Attachment(s) present."
                elif message['embeds']:
                    message_content = "Embed(s) present."
                else:
                    message_content = "Empty message."
            detailed_messages.append({
                "author": message['author']['username'],
                "content": message_content,
                "timestamp": message['timestamp']
            })

        response = f'The last {n} messages in descending orrder are: \n' + str(detailed_messages)
        return response
        
    def set_tools(self):
        other_tools = [self._tools['Calendar'], self.extend_tools['CSV'], self.extend_tools['Read Discord Messages'], self.extend_tools['Send Discord Message']]
        #return other_tools
        return self.gmail_tools + other_tools 
    
        #return self.gmail_tools
    def set_memory(self):
        memory=ConversationBufferWindowMemory(k=2)
        return memory
    def set_prompt_template(self):
        return ''
    
    def initialize_agent(self):
        llm = self.llm #OpenAI(temperature=0)
        self.agent = initialize_agent(
        tools= self.tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        #return_intermediate_steps=True
        memory = self.memory,
        verbose=True 
        )
        logging.debug(f'agent is {self.agent}')
        logging.debug(f'agent template {self.agent.agent.llm_chain.prompt}')

    def run_agent(self, input_prompt: str) -> str:
        '''
        prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
        suffix = """Question: {task} {agent_scratchpad}"""
        prompt = ZeroShotAgent.create_prompt(
                                            self.tools, 
                                            prefix=prefix, 
                                            suffix=suffix, 
                                            input_variables=["objective", "task", "context","agent_scratchpad"]
        )
        input_objective = {"objective":input_prompt}
        bagi =  BabyAGI.from_llm(llm=self.llm, 
                                 prompt_template=self.prompt_template, 
                                 tools=self.tools, 
                                 max_iterations=max_iterations) 

       



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
         '''
        return self.agent.run(input_prompt) 

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
from baseAssistantAgent import BaseAssistantAgent, CustomPromptTemplate, CustomOutputParser
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



class MedicalAssistantAgent(BaseAssistantAgent):
    def __init__(self):
        super().__init__()
    
    def run_agent(self, input_prompt: str) -> str:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        tool_names = [tool.name for tool in self.tools]

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser= self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        # Set agent executor
        # Agent Executors take an agent and tools and use the agent to decide which tools to call and in what order.
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=self.tools,
                                                    verbose=True,
                                                    memory=self.memory)
        agent_output = agent_executor.run(input_prompt)
        self.process = agent_output 
        print("debugging agent output: ", agent_output)
        final_answer = agent_output.split("Final Answer:")[-1].strip()
        return final_answer

    def set_tools(self):
        
        return [self._tools['Web Search'], self._tools['Search WebMD']]
    

    def set_prompt_template(self):
        template = """Answer the following questions as best you can, but speaking as compasionate medical professional. You have access to the following tools:

                        {tools}

                        Use the following format:

                        Question: the input question you must answer
                        Thought: you should always think about what to do
                        Action: the action to take, should be one of [{tool_names}]
                        Action Input: the input to the action
                        Observation: the result of the action
                        ... (this Thought/Action/Action Input/Observation can repeat N times)
                        Thought: I now know the final answer
                        Final Answer: the final answer to the original input question

                        Begin! Remember to answer as a compansionate medical professional when giving your final answer.
                        Previous conversation history:
                        {history}
                        Question: {input}
                        {agent_scratchpad}"""
        return CustomPromptTemplate(
            template=template,
            tools=self.tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps", "history"]
        )

    def set_memory(self):
        memory=ConversationBufferWindowMemory(k=5)
        return memory
    
    def get_summary(self):
        return self.memory


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
from langchain_experimental.pal_chain import PALChain
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from abc import ABC, abstractmethod
from datetime import date
from pydantic import BaseModel, Field

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


class BaseAssistantAgent(ABC):
    def __init__(self):
        self._tools = self.build_tools()
        self.tools = self.set_tools()
        self.prompt_template = self.set_prompt_template()
        self.llm = OpenAI(temperature=0)
        self.memory = self.set_memory()
        self.output_parser = CustomOutputParser()

    def build_tools(self):
            tools = {
                #general web search tool
                "Web Search": Tool(
                    #
                    name = "Web Search",
                    func =  DuckDuckGoSearchRun().run,
                    description="useful for when you need to answer questions about current events",
                ),
                "Search WebMD" : Tool(
                    name = "Search WebMD",
                    func= self._search_webmd,
                    description="useful for when you need to answer medical and pharmalogical questions"
                    ),
                "Today Date": Tool(
                    name = "Today Date",
                    func= self._todays_date,
                    description="useful for when you need to know the current date",
                    ),
                "PAL": Tool(
                    name = "PAL",
                    func= self._pal,
                    description="useful for when you need to answer questions about math or word problems or date comparisons",
                    )
            }
            return tools
    
    def _search_webmd(self, input):
        search = DuckDuckGoSearchRun()
        search_results = search.run(f"site:webmd.com {input}")
        return search_results
    
    def _todays_date(self, input):
        return date.today().strftime("%B %d, %Y")
    
    def _pal(self, input):
        llm = OpenAI(temperature=0, verbose=True)
        pal_chain = PALChain.from_math_prompt(llm, verbose=True)
        return pal_chain.run(input)
    
    @abstractmethod
    def set_tools(self):
        pass
    @abstractmethod
    def set_memory(self):
        pass
    @abstractmethod
    def set_prompt_template(self):
        pass
    @abstractmethod
    def run_agent(self, input_prompt: str) -> str:
        pass


class CustomOutputParser(AgentOutputParser):
    '''
    The output parser is responsible for parsing the LLM output into AgentAction and AgentFinish. This usually depends heavily on the prompt used.
    This is where you can change the parsing to do retries, handle whitespace, etc
    '''
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class CustomPromptTemplate(StringPromptTemplate):
    '''
    This instructs the agent on what to do. Generally, the template should incorporate:

    tools: which tools the agent has access and how and when to call them.

    intermediate_steps: These are tuples of previous (AgentAction, Observation) pairs. These are generally not passed directly to the model, but the prompt template formats them in a specific way.

    input: generic user input
    '''
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
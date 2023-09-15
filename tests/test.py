from langchain.agents.agent_toolkits import GmailToolkit

toolkit = GmailToolkit()

tools = toolkit.get_tools()
print(tools) 



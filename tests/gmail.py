
#self.handle_token('email')
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain.agents.agent_toolkits import GmailToolkit
credentials = get_gmail_credentials(
    token_file="token_mail.json",
    scopes=["https://mail.google.com/"], #"https://www.googleapis.com/auth/calendar"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)
#toolkit = GmailToolkit()
print(toolkit.get_tools())
tools = toolkit.get_tools()

tols_dict = {}
for t in tools:
    tols_dict[t.name] = t

print('----------------')
print(tols_dict['send_gmail_message'])  
print('----------------')
print(tols_dict['create_gmail_draft'])  
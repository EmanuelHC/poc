
import discord
from discord.ext import commands
import requests
import os
import dotenv

dotenv.load_dotenv()

CHANNEL_ID = '1152713683185704980'
TOKEN = os.environ.get("DISCORD_TOKEN", "")
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_ID", "")
BASE_URL = "https://discord.com/api/v10"
GUILD_ID = os.environ.get("DISCORD_GUILD_ID", "")

HEADERS = {
    "Authorization": f"Bot {TOKEN}",
    "Content-Type": "application/json",
}

print(f'Discord token: {TOKEN}')

def read_last_n_messages(n=10):
    response = requests.get(f"{BASE_URL}/channels/{CHANNEL_ID}/messages?limit={n}", headers=HEADERS)
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

    return detailed_messages

def send_message(content):
    """Send a message to the Discord channel using a webhook."""
    data = {"content": content}
    response = requests.post(WEBHOOK_URL, json=data)
    return response.status_code == 204  # 204 means the message was sent successfully

# Example usage:
print(read_last_n_messages(5))
#send_message("Hello from the Python script!")



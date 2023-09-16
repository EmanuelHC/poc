import google.oauth2.credentials
import google_auth_oauthlib.flow
from googleapiclient.discovery import build

# Load your credentials directly from the token.json file

#

# If the credentials are expired, refresh them
def refresh_google_token(token_file_name:str = 'token.json'):
    credentials = google.oauth2.credentials.Credentials.from_authorized_user_file(token_file_name)
    if credentials.expired:
        credentials.refresh(google.auth.transport.requests.Request())

    # Save the new token
    with open(token_file_name, 'w') as token_file:
        token_file.write(credentials.to_json())

    print(f"Token refreshed and saved to {token_file_name}")

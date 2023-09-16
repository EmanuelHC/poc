import openai
import streamlit as st
import subprocess
import dotenv 
from utils.utils import CaptureLogs, stdout_stream, stderr_stream
from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder
from agents.medicalAssistantAgent import MedicalAssistantAgent
from agents.generalAssistantAgent import GeneralAssistanAgent
from  agents.skillfulAssistantAgent import  SkillfulAssistantAgent
import numpy as np
#openai.api_base = "https://api.openai.com"
import google.oauth2.credentials
import google_auth_oauthlib.flow
from googleapiclient.discovery import build
import pandas as pd
from io import StringIO


SCOPES = ['https://mail.google.com/', 'https://www.googleapis.com/auth/calendar']

from refresh_google_token import refresh_google_token
token_file_name = 'token_mail.json'
credentials = google.oauth2.credentials.Credentials.from_authorized_user_file(token_file_name)
refresh_google_token(token_file_name) 



MEDICAL_ASSITANT_TAG = 'Skillful Medical Assistant'
GENERAL_ASSITANT_TAG = "Skillful General Assistant"
#WORK_ASSISTANT_TAG = "Skillful Work Assistant"
SKILLFUL_ASSISTANT_TAG = "Skillful AI Assistant"
# Set page config at the top
st.set_page_config(layout="wide")
# Add Streamlit customization
st.markdown("""
<style>
    /* This will set the main background color */
    body, .main {
        background-color: linear-gradient(90deg, #140012 11.32%, #00003A 50.08%, #000002 90.49%); !important;
    }

    /* This will change the button style */
    .stButton>button {
        background-color: linear-gradient(95deg, #F05F47 0.73%, #DE3593 92.39%);
;
        color: white;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)




def get_selected_agent(assistant):
    if assistant == MEDICAL_ASSITANT_TAG:
        return MedicalAssistantAgent()
    elif assistant == GENERAL_ASSITANT_TAG:
        return GeneralAssistanAgent()

    elif assistant == SKILLFUL_ASSISTANT_TAG:
        return SkillfulAssistantAgent() 
        #return WorkAssistantAgent()

def main():
    st.title(":violet[Skillful AI POC]")

   

    #option = st.selectbox('Select a model option from the list:', ('gorilla-7b-hf-v1', "gorilla-mpt-7b-hf-v0","gorilla-7b-hf-v1-ggml"))

    
    assistant_images = {
        GENERAL_ASSITANT_TAG: "assets/general_assistant_01.jpeg",
        MEDICAL_ASSITANT_TAG: "assets/medical_assistant_03.jpeg",
     
        SKILLFUL_ASSISTANT_TAG: "assets/work_assistant_02.jpeg"
    }

    # Create two columns for layout
    col1, col2, col3  = st.columns([1,1, 1])

    with col1:
        assistant = st.selectbox('Select an assistant option from the list:', ( GENERAL_ASSITANT_TAG,
                                                                                MEDICAL_ASSITANT_TAG, 
                                                                            SKILLFUL_ASSISTANT_TAG,))
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        # Check if a file has been uploaded
        if uploaded_file:
        # Save the uploaded file to a desired location
            with open("uploaded_file.csv", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Read the file in binary mode and decode with error handling
            with open('uploaded_file.csv', 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

            # Convert the decoded content to a pandas DataFrame
            df = pd.read_csv(StringIO(content))

            # Optional: Remove rows that contain the Unicode replacement character
            df = df[~df.applymap(lambda x: '�' in str(x)).any(axis=1)]

            # Drop rows with NaN values
            df.dropna(inplace=True)

            # Save the cleaned data to a new CSV file
            df.to_csv('cleaned_file.csv', index=False)


            st.success("File uploaded and saved!")

    with col2:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="7F00FF",
            #icon_name="user",
            icon_size="6x",
            energy_threshold=(-1.0, 1.0),
            pause_threshold=10.0,

          
        )
   

        if audio_bytes:
            #st.audio(note_la, sample_rate=sample_rate,format="audio/wav")
            st.audio(audio_bytes, format="audio/wav")
        
       

    with col3:
        # Set initial size of the image
        image_size = 'auto'
        
        # If an option has been selected, upscale the image a bit
        if assistant:
            image_size = '75%'
        
        st.image(assistant_images[assistant], width=200,)

    input_prompt = st.text_area("Enter your prompt below:")
    
    if st.button("Run"):
        if len(input_prompt) > 0:
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.subheader("Output")
                agent =  get_selected_agent(assistant)
                with CaptureLogs():
                    try:
                        result = agent.run_agent(input_prompt)
                    except Exception as e:
                        result = f"Proccess not finished succesfully due to: {e}" 
                st.write(result)

            
            if 'previous_output' not in st.session_state:
                st.session_state.previous_output =   ''
            with col2:
                # pass        
                st.subheader("Process")
                current_output = stdout_stream.getvalue()
                # Extract the new content added since the last iteration
                new_output = current_output[len(st.session_state.previous_output):]
                # Display the new content in Streamlit
                st.write(new_output)
                # Update the session state variable for the next iteration
                st.session_state.previous_output = current_output
            with col3:
                st.subheader("Memory")
                st.write( agent.memory)
                
                

        
if __name__ == "__main__":
    main()
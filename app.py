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
from utils.tts import run_tts
from utils.utils import  get_base64_of_bin_file
import os 
import io 
import sys 
import tempfile
SCOPES = ['https://mail.google.com/', 'https://www.googleapis.com/auth/calendar']

from refresh_google_token import refresh_google_token
token_file_name = 'token_mail.json'
credentials = google.oauth2.credentials.Credentials.from_authorized_user_file(token_file_name)
refresh_google_token(token_file_name) 

import base64

def play_audio_autoplay(audio_bytes):
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio controls autoplay><source src="data:audio/wav;base64,{audio_b64}" type="audio/wav"></audio>'
    st.markdown(audio_tag, unsafe_allow_html=True)

openai.api_key = os.environ.get("OPENAI_API_KEY", "")
MEDICAL_ASSITANT_TAG = 'Skillful Medical Assistant'
GENERAL_ASSITANT_TAG = "Skillful General Assistant"
#WORK_ASSISTANT_TAG = "Skillful Work Assistant"
SKILLFUL_ASSISTANT_TAG = "Skillful AI Assistant"

OUTPUT_CHAIN_FILE_NAME = 'agent_execution_chain.txt'
# Set page config at the top
st.set_page_config(layout="wide")
bin_background_image = get_base64_of_bin_file('assets/back3.png')
logo_image_base64 = get_base64_of_bin_file('assets/logo2c.png')

#print(bin_background_image)
# Add Streamlit customization
page_bg_img = f"""
<style>
    /* Set the main background color and style */
    body, .main {{
        background-image: url("data:image/png;base64,{bin_background_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Arial', sans-serif;
        color: #333;
    }}

    /* Add the logo to the top right corner and set its size */
    body:before {{
        content: url("data:image/png;base64,{logo_image_base64}");
        position: absolute;
        top: 50px;
        right: 150px;
        width: 10px;  /* Adjust this value to set the width of the logo */
        height: auto; /* This will maintain the aspect ratio */
        z-index: 1000; /* Ensure the logo is above other elements */
    }}

    /* Style the Streamlit button */
    .stButton>button {{
        background-color: #F05F47;
        background-image: linear-gradient(96deg, #8E8FF5 1.87%, #030046 50.61%, #02012D 97.37%);
        color: white;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        border-radius: 4px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }}

    .stButton>button:hover {{
        background-color: #DE3593;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }}

    .stButton>button:active {{
        transform: translateY(0);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)




def get_selected_agent(assistant):
    if assistant == MEDICAL_ASSITANT_TAG:
        return MedicalAssistantAgent()
    elif assistant == GENERAL_ASSITANT_TAG:
        return GeneralAssistanAgent()

    elif assistant == SKILLFUL_ASSISTANT_TAG:
        return SkillfulAssistantAgent() 
        #return WorkAssistantAgent()

def main():

    st.markdown("""
    <div style="font-size: 4.5em; font-weight: bold; background: linear-gradient(96deg, #8E8FF5 1.87%, #030046 50.61%, #02012D 97.37%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;">
        Skillful AI POC
    </div>
    """, unsafe_allow_html=True)
   

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
        if 'transcription_complete' not in st.session_state:
            st.session_state.transcription_complete = False
        if 'transcribed_text' not in st.session_state:
            st.session_state.transcribed_text = ""
        if 'input_prompt' not in st.session_state:
            st.session_state.input_prompt = ""
        col1, col2, col3 = st.columns([1, 6, 1])  
        with col1:
            audio_bytes = audio_recorder(
                text="",
                #recording_color="#e8b62c",
                neutral_color="#6060F2",
                #icon_name="square",
                icon_size="2x",
                energy_threshold=(-1.0, 1.0),
                pause_threshold=10.0,

            
            )
        with col2:
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                # Save audio_bytes to a temporary file
                temp_audio_filename = "input_prompt.wav"
                with open(temp_audio_filename, "wb") as temp_audio_file:
                    temp_audio_file.write(audio_bytes)
                
                # Transcribe the audio using the saved file
                with open(temp_audio_filename, "rb") as audio_file:
                    transcript = openai.Audio.transcribe("whisper-1", audio_file)
                    transcribed_text = transcript.text
                    st.session_state.transcribed_text = transcribed_text
                    st.write(transcribed_text)

                # Save the transcribed text to a file
                with open("transcription.txt", "w") as text_file:
                    text_file.write(transcribed_text)

                st.success("Transcription saved to transcription.txt!")
                st.session_state.input_prompt = transcribed_text
                st.session_state.transcription_complete = True


        with col3:
            pass    
       

    with col3:
        # Set initial size of the image
        image_size = 'auto'
        
        # If an option has been selected, upscale the image a bit
        if assistant:
            image_size = '75%'
        
        st.image(assistant_images[assistant], width=200,)

    #input_prompt = st.text_area("Enter your prompt below:")
    input_prompt = st.text_area("Enter your prompt below:", value=st.session_state.input_prompt if st.session_state.transcription_complete else "")

    
    if st.button("Run") or st.session_state.transcription_complete:
    #if st.button("Run"):
        # Delete existing temp file
        if os.path.exists(OUTPUT_CHAIN_FILE_NAME):
            os.remove(OUTPUT_CHAIN_FILE_NAME)

        # Create a new temp file
       
      

        if len(input_prompt) > 0:
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.subheader("Output")
                agent =  get_selected_agent(assistant)
                try:
                    # Redirect stdout to file
                    with CaptureLogs():
                    #with open(OUTPUT_CHAIN_FILE_NAME, 'w') as temp_file:
                        #original_stdout = sys.stdout
                        #sys.stdout = temp_file
                        result = agent.run_agent(input_prompt)
                        #sys.stdout = original_stdout

                    audio_path = run_tts(result)
                    with open(audio_path, 'rb') as f:
                        speech_result = f.read()
                except Exception as e:
                    result = f"Proccess not finished succesfully due to: {e}" 
                    speech_result = None

                st.markdown("<span style='color: white; font-size: 20px;'>{}</span>".format(result), unsafe_allow_html=True)
                if speech_result:
                    play_audio_autoplay(speech_result)
            if 'previous_output' not in st.session_state:
                st.session_state.previous_output =   ''
            with col2:
                st.subheader("Process")
                current_output = stdout_stream.getvalue()
                # Extract the new content added since the last iteration
                new_output = current_output[len(st.session_state.previous_output):]
                # Display the new content in Streamlit
                #st.write(new_output)
                st.markdown("<span style='color: white; font-size: 20px;'>{}</span>".format(new_output), unsafe_allow_html=True)
                # Update the session state variable for the next iteration
                st.session_state.previous_output = current_output
                
                #with open(OUTPUT_CHAIN_FILE_NAME, 'r') as f:
                    #content = f.readlines()
                    # Remove lines that start with #
                    #cleaned_content = ''.join([line for line in content if not line.strip().startswith('#')])
                    #st.markdown("<span style='color: white; font-size: 20px;'>{}</span>".format(cleaned_content), unsafe_allow_html=True)
                    #content = f.read()
                    #st.markdown("<span style='color: white; font-size: 20px;'>{}</span>".format(content), unsafe_allow_html=True)
                
            with col3:
                st.subheader("Memory")
                memories = agent.memory.load_memory_variables({})
                #st.write(agent.memory)
                st.write(memories)
                

        
if __name__ == "__main__":
    main()
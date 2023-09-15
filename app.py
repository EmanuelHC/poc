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
        MEDICAL_ASSITANT_TAG: "assets/medical_assistant_03.jpeg",
        GENERAL_ASSITANT_TAG: "assets/general_assistant_01.jpeg",
        SKILLFUL_ASSISTANT_TAG: "assets/work_assistant_02.jpeg"
    }

    # Create two columns for layout
    col1, col2, col3  = st.columns([1,1, 1])

    with col1:
        assistant = st.selectbox('Select an assistant option from the list:', (MEDICAL_ASSITANT_TAG, 
                                                                            GENERAL_ASSITANT_TAG,
                                                                            SKILLFUL_ASSISTANT_TAG,))


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
                    result = agent.run_agent(input_prompt) 
                st.write(result)
                
            with col2:
                # pass        
                st.subheader("Process")
                # Display captured logs in Streamlit
                st.write( stdout_stream.getvalue())
            with col3:
                st.subheader("Memory")
                st.write( agent.memory)
                
                

        
if __name__ == "__main__":
    main()
import sys
import io
import streamlit as st
import re
import inflect
from dateutil import parser
import base64
import PyPDF2 

def get_log_streams():
    return io.StringIO(), io.StringIO()

stdout_stream, stderr_stream = get_log_streams()

class CaptureLogs:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = stdout_stream
        sys.stderr = stderr_stream

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr



@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

#print(get_base64_of_bin_file('assets/back.png'))


def convert_to_words(text):
    p = inflect.engine()

    # Convert dates
    date_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})'
    text = re.sub(date_pattern, lambda x: '/'.join([p.number_to_words(i) for i in x.group().split('/')]), text)

    # Convert decimal numbers
    decimal_pattern = r'(\d+)\.(\d+)'
    text = re.sub(decimal_pattern, lambda x: p.number_to_words(int(x.group(1))) + " point " + ' '.join([p.number_to_words(int(i)) for i in x.group(2)]), text)

    # Convert whole numbers
    number_pattern = r'(?<!\d\.)(\d+)(?!\.\d)'
    text = re.sub(number_pattern, lambda x: p.number_to_words(int(x.group())), text)

    return text

def extract_text_from_pdf(uploaded_file):
    """Extract text from the uploaded PDF file."""
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page_num).extractText()
    return text

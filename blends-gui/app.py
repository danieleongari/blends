#try:
#    from blends import Blend
#except ImportError:
#    import sys
#    sys.path.append('../')    
#    from blends import Blend

from dotenv import load_dotenv, find_dotenv #python-dotenv
import openai

import os

import streamlit as st
import pandas as pd

import logging

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

@st.cache_resource
def start_api():
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.getenv('OPENAI_API_KEY')
    logging.info(f"api_key acquired and cached")

def main():

    start_api()

    # ----- Header of the app -----
    st.title("BlenDS")
    st.write("An intuitive specification of the design space for blends of components")

    # ----- Write questions separated by a new line -----
    st.header("What do you want to create?")
    question = st.text_area("Questions", value = "")

    # Initialize session_state if not exists
    logging.info(f'session: {st.session_state.keys()}')
    if 'res' not in st.session_state:
        st.session_state.res = []
        logging.info(f'chat history initialized')
    logging.info(f"chat history length: {len(st.session_state.res)}")

    if len(st.session_state.res)>0:
        st.header("Answer")
        anss = st.session_state.res[-1].split('\n\n')
        for ans in anss:
            print('ans:',ans)
            #if len(ans)>4:
            #    if is_valid_table(ans): # this does not work
            #        logging.debug(f"This is a table. Displaying it")
            #        df = parse_text_to_dataframe(ans)
            #        # Display the DataFrame as a table
            #        st.table(df)
            #    else:
            #        st.write(f"{ans}")

    # Button to clear the list
    if st.button("Clear chat history"):
        st.session_state.res.clear()
        st.write("Chat history cleared")

if __name__ == "__main__":
    main()

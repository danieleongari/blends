try:
    from blends import Blend
except ImportError:
    import sys
    sys.path.append('../')    
    from blends.viz import get_graph
    from blends.base import dict_to_blend
    from blends.sample import get_samples

from dotenv import load_dotenv, find_dotenv #python-dotenv
import openai

from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager  
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import os
import re

import streamlit as st
import pandas as pd

import logging

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG)

@st.cache_resource
def start_api():
    _ = load_dotenv(find_dotenv()) # needs .env with openai passkey
    openai.api_key = os.getenv('OPENAI_API_KEY')
    logging.info(f"api_key acquired and cached")

#If you should need to return a table, the format must be similar to
#this example```\n\nIndex\tColA\tColB\nColC\n1\t1.2\t1.5\t1.8\n2\t1.3\t1.8\t1.9\n\n```."

PROMPT_TEMPLATE = """
You are an expert research lab assistant. If provided, use the chat history to refine your answers.
Your task is to generate a dictionary that contains the hierarchical structure for specifying blends of components
the user is looking to create. This is an example:
```
{example}
```
This is how you must interpret the dictionary.
You start from the root and assign a "name" to the blend of the component and a brief "description".
The blend can have one or more "children" that can be recursive. 
These are the components of the blend that make up the blend. 
Each child component can have its own name, description, and quantity constraints.
This is the question:
```
{question}
```
This is the chat history:
```
{history}
``` 
Answer:
""".strip()

BLEND_EXAMPLE = {
    "name": "myRootBlend",
    "description": "an example of a Blend",
    "qmin": 1.0,
    "qmax": 1.0,
    "children": [
        {
            "name": "Solvent",
            "description": "Choose only one of the 3 following solvents",
            "qmin": 0.2,
            "qmax": 0.3,
            "cmax": 1,
            "children": [
                {
                    "name": "Solvent-1",
                    "description": "xxx"
                },
                {
                    "name": "Solvent-2",
                    "description": "yyy"
                },
                {
                    "name": "Solvent-3",
                    "description": "zzz"
                }
            ]
        },
        {
            "name": "Blend-A",
            "description": "Mix the following A components",
            "qmin": 0.4,
            "qmax": 0.7,
            "children": [
                {
                    "name": "Comp-A1",
                    "description": "xxx"
                },
                {
                    "name": "Comp-A2",
                    "description": "yyy"
                },
                {
                    "name": "Comp-A3",
                    "description": "zzz"
                }
            ]
        },
        {
            "name": "Component-B",
            "description": "zzz",
            "qmin": 0.1,
            "qmax": 0.3
        },
        {
            "name": "Component-C",
            "description": "zzz",
            "qmin": 0.0,
            "qmax": 0.2,
            "cmax": 1,
            "children": [
                {
                    "name": "Comp-C1",
                    "description": "xxx",
                    "children": [   
                        {
                            "name": "Comp-C1-1",
                            "description": "xxx"
                        },
                        {
                            "name": "Comp-C1-2",
                            "description": "yyy"
                        }
                    ]
                },
                {
                    "name": "Comp-C2",
                    "description": "yyy"
                },
            ]
        }        

    ]

}

def query_ai(question, example, history=''):

    model = ChatOpenAI(temperature=0, model='gpt-4-0125-preview')
      
    prompt = PromptTemplate(input_variables=['question', 'example', 'history'], template=PROMPT_TEMPLATE)
    memory = ConversationBufferWindowMemory(memory_key='history', input_key='question', k=6)
    chain = LLMChain(llm=model, memory=memory, prompt=prompt, verbose=True)
    logging.debug(f"prompt: {chain.prompt.template}")

    return chain.run(question=question, example=example, history=history)

def main():

    start_api()

    # ----- Header of the app -----
    st.title("BlenDS")
    st.write("An intuitive specification of the design space for blends of components")

    # Initialize session_state if not exists
    logging.info(f'session: {st.session_state.keys()}')
    if 'history' not in st.session_state:
        st.session_state.history = []
        logging.info(f'chat history initialized')
    logging.info(st.session_state)
    logging.info(f"chat history length: {len(st.session_state.history)}")

    # ----- Write questions separated by a new line -----
    st.header("What do you want to create?")
    question = st.text_area("", value = "")

    with st.spinner('Generating results...'):
        res_str = query_ai(question, BLEND_EXAMPLE, '')

    st.header("Answer")
    st.info(res_str)

    logging.debug(f'Result: {res_str}')

    # Define the pattern to match
    res_pattern = r'```json\s*({[\s\S]*?})\s*```'
    # Use re.search to find the first occurrence of the pattern in the string
    match = re.search(res_pattern, res_str)

    # If a match is found, extract the captured group
    if match:
        res_str = match.group(1)
    else:
        logging.error("No match found")

    res_dict = eval(res_str)

    st.header("Visualize")
    res_blend = dict_to_blend(res_dict)
    res_graph = get_graph(res_blend)
    st.graphviz_chart(res_graph)

    st.header("Download")
    with st.spinner('Generating trial...'):
        df = get_samples(res_blend, nsamples=1000, verbose=True)
        st.dataframe(df)


    # Button to clear the list - TODO
    #if st.button("Clear chat history"):
    #    st.session_state.history.clear()
    #    st.write("Chat history cleared")

if __name__ == "__main__":
    main()

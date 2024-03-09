import os
import streamlit as st

from langchain.llms import OpenAI
from apikey import apikey
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ['OPENAI_API_KEY'] = apikey



#APP
st.title('⛓️🤖 【L】【L】【M】')
prompt = st.text_input("Your Prompt Below")

############
#Prompt Template
task_template = PromptTemplate(
    input_variables=['projectName'],
    template='Give me the name of all the files needed for the {projectName} and don\'t give headlines in your output and don\'t include any explanations in your responses and give good spacing'
)

code_template = PromptTemplate(
    input_variables=['fileDetails'],
    template='Give me full code in python programming language for : {fileDetails} and don\'t give headlines in your output and don\'t include any explanations in your responses and give good spacing and don\'t include any comments'
)

#Memory

memory= ConversationBufferMemory(input_key='projectName', memory_key='chat_history')



##########

#llms
llm = OpenAI(temperature=0)
task_chain = LLMChain(llm=llm, prompt=task_template, verbose=True, output_key='fileDetails')
code_chain = LLMChain(llm = llm, prompt=code_template, verbose=True, output_key='code')
sequential_chain = SequentialChain(chains=[task_chain, code_chain], input_variables=['projectName'], output_variables=['fileDetails', 'code'], verbose=True)



#Display o/p
if prompt:
    response = sequential_chain({'projectName':prompt})
    st.write(response['fileDetails'])
    st.write(response['code'])
    with st.expander('Message History'):
        st.info(memory.buffer)



## topic-ProjectName
## title-fileDetails
## script-code
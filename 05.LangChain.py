from langchain_openai import AzureChatOpenAI  
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage 

from dotenv import load_dotenv
import os

load_dotenv()

MODEL_DEPLOYMENT_NAME = "gpt-4.1"

llm = AzureChatOpenAI(
        deployment_name=MODEL_DEPLOYMENT_NAME,                        
        temperature=1,                        
)  

messages = [
         SystemMessage(content='당신은 업무 계획을 세워주는 업무 플래너 머신입니다. 사용자의 업무를 입력 받으면 이를 위한 계획을 작성합니다.'),     
         HumanMessage(content='신입사원 교육을 해야됩니다.') 
]  

# answer = llm.invoke(messages) 
# print(answer.content)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('/workspaces/LLMTest/Owners_Manual.pdf') 
pages = loader.load_and_split()

print(pages[0].page_content)
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = 'text-embedding-3-large'
persist_directory = './chroma_db'

embedding_model = AzureOpenAIEmbeddings(
                        azure_deployment=EMBEDDING_MODEL_NAME, 
                        chunk_size=1000)

# 저장된 데이터베이스 불러오기
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

from langchain_classic.chains import RetrievalQA
from re import search  
from langchain_openai import AzureChatOpenAI  

MODEL_DEPLOYMENT_NAME = "gpt-4.1" # 실제 Azure 배포명을 입력하세요

llm = AzureChatOpenAI(
        deployment_name=MODEL_DEPLOYMENT_NAME,                        
        temperature=1,                        
)  

qa = RetrievalQA.from_chain_type(
         llm=llm,
         chain_type='stuff',
         retriever=db.as_retriever(
                 search_type='mmr',
                 search_kwargs={'k': 3,'fetch_k': 10}             ),
         return_source_documents=True         )

query = '실내 온도 조절 장치의 전기 공급은?' 
result = qa.invoke(query)
print("Answer: ", result['result'])
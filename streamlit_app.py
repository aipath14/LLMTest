import os
from dotenv import load_dotenv

# Streamlit UI
import streamlit as st

# Try recommended Chroma import with fallback
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_classic.chains import RetrievalQA

load_dotenv()

MODEL_DEPLOYMENT_NAME = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4.1")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

@st.cache_resource
def create_qa():
    llm = AzureChatOpenAI(deployment_name=MODEL_DEPLOYMENT_NAME, temperature=1)
    embedding_model = AzureOpenAIEmbeddings(azure_deployment=EMBEDDING_MODEL_NAME, chunk_size=1000)

    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}),
        return_source_documents=True,
    )
    return qa


def extract_answer_and_sources(result):
    # Support multiple return formats
    answer = None
    sources = None
    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or result.get("output") or result.get("text")
        sources = result.get("source_documents") or result.get("source_documents", None)
    else:
        answer = str(result)
    return answer, sources


# --- Streamlit UI ---
st.set_page_config(page_title="Vector Search Chat", layout="wide")
st.title("Vector Search Chat")
st.write("간단한 Retrieval QA 채팅 인터페이스입니다. 질문을 입력하면 벡터 DB에서 관련 문서를 찾아 답변합니다.")

if "history" not in st.session_state:
    st.session_state.history = []

qa = create_qa()

with st.form(key="query_form"):
    query = st.text_input("질문을 입력하세요", "", key="query_input")
    submitted = st.form_submit_button("전송")

if submitted and query:
    try:
        # Prefer .invoke if available, else try .run
        if hasattr(qa, "invoke"):
            result = qa.invoke(query)
        elif hasattr(qa, "run"):
            result = qa.run(query)
        else:
            result = qa(query)

        answer, sources = extract_answer_and_sources(result)

    except Exception as e:
        answer = f"에러가 발생했습니다: {e}"
        sources = None

    st.session_state.history.append({"query": query, "answer": answer, "sources": sources})

# Display chat history (most recent last)
for turn in st.session_state.history:
    st.markdown("**사용자:** " + turn["query"])
    st.markdown("**어시스턴트:** " + (turn["answer"] or "(응답 없음)"))
    if turn.get("sources"):
        st.markdown("**출처 문서:**")
        for i, doc in enumerate(turn["sources"][:5], start=1):
            # doc may be a dict-like or object with attributes
            content = getattr(doc, "page_content", None) or doc.get("page_content") if isinstance(doc, dict) else str(doc)
            metadata = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else None)
            excerpt = content[:800] + ("..." if content and len(content) > 800 else "") if content else "(본문 없음)"
            st.markdown(f"- 문서 {i}: {excerpt}")
            if metadata:
                st.caption(f"메타데이터: {metadata}")

st.button("대화 초기화", on_click=lambda: st.session_state.clear())

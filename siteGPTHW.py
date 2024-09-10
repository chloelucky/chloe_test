import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# 앱 설정
st.set_page_config(page_title="Cloudflare SiteGPT", page_icon="☁️")

# Sidebar: API 키 입력
with st.sidebar:
    st.title("Cloudflare SiteGPT")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    
    # GitHub 리포지토리 링크 추가
    st.markdown("[GitHub Repo](https://github.com/your-repo)")

# API 키가 없다면 경고 표시
if not openai_api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar.")
    st.stop()

# Sitemap URL 및 문서 로드
sitemap_url = "https://developers.cloudflare.com/sitemap-0.xml"

@st.cache_data(show_spinner="Loading Cloudflare documentation...")
def load_cloudflare_docs():
    # 사이트맵 로더 설정
    loader = SitemapLoader(sitemap_url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = loader.load_and_split(text_splitter=splitter)
    
    # 문서 임베딩 및 벡터 스토어 생성
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# 문서 로드 및 검색 기능 설정
retriever = load_cloudflare_docs()

# 사용자 입력: 질문 받기
query = st.text_input("Ask a question about Cloudflare products:")

# 답변 처리
if query:
    # OpenAI 모델과 검색 체인 설정
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")
    
    # 질문에 대한 답변 생성
    with st.spinner("Searching the Cloudflare documentation..."):
        response = qa_chain.run(query)
    
    # 결과 출력
    st.write("### Answer:")
    st.write(response)

# 문서 로드 및 문제 해결
st.write("""
    ## Cloudflare Documentation QA Bot
    
    This chatbot provides answers to questions about the following Cloudflare products:
    - [AI Gateway](https://developers.cloudflare.com/ai-gateway/)
    - [Cloudflare Vectorize](https://developers.cloudflare.com/vectorize/)
    - [Workers AI](https://developers.cloudflare.com/workers-ai/)
    
    The data is pulled directly from Cloudflare's official documentation.
""")
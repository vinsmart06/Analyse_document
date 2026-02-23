import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

#load_dotenv()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Enterprise Document Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Enterprise Document Analytics Chatbot")
st.caption("Powered by AI LLM + LangChain: Developed by Vineet")

# ----------------------------
# SESSION STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ----------------------------
# FILE UPLOAD
# ----------------------------
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or DOCX",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# ----------------------------
# DOCUMENT LOADING
# ----------------------------
def load_documents(files):
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = Docx2txtLoader(tmp_path)

        docs = loader.load()
        documents.extend(docs)

    return documents


def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    return FAISS.from_documents(split_docs, embeddings)


# ----------------------------
# BUILD VECTORSTORE
# ----------------------------
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Processing documents..."):
        docs = load_documents(uploaded_files)
        st.session_state.vectorstore = create_vectorstore(docs)
    st.sidebar.success("Documents processed successfully ✅")

# ----------------------------
# CHAT SYSTEM
# ----------------------------
if st.session_state.vectorstore:

    llm = ChatGoogleGenerativeAI(
        model= "gemini-3-flash-preview",
        temperature=0,
        streaming=True  # 🔥 ENABLE STREAMING
    )

    # Reformulation prompt (history-aware retriever)
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and latest user question, "
         "rephrase the question to be standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])


    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])



#    document_chain = create_stuff_documents_chain(llm, qa_prompt)

#    rag_chain = create_retrieval_chain(
#        history_aware_retriever,
#        document_chain
#    )



    user_input = st.chat_input("Ask something about your documents...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        with st.chat_message("user"):
            st.markdown(user_input)
        relevant_docs = st.session_state.vectorstore.similarity_search(user_input,K=5)
 #       relevant_docs = retriever.retrieve(user_input)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using ONLY the provided context. "
         "If answer not found, say 'I don't know.'"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("system", "Context:\n{context}")
        ])

        formatted_input = qa_prompt.format_prompt(
            input=user_input,
            context=context,
            chat_history=st.session_state.chat_history
        ).to_messages()

                # ----------------------------
        # Stream LLM response
        # ----------------------------
        full_response = ""
        message_placeholder = st.empty()

        with st.spinner("Thinking..."):
           for token in llm.stream(formatted_input):
                full_response += token.text
                message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        st.session_state.chat_history.append(
            AIMessage(content=full_response)
        )

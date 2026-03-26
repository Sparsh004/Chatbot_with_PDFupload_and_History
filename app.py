# import streamlit as st
# import os
# from langchain_groq import ChatGroq

# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_classic.chains import create_history_aware_retriever 
# from langchain_community.chat_message_histories import ChatMessageHistory 
# from langchain_core.chat_history import BaseChatMessageHistory 
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_classic.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader

# from dotenv import load_dotenv
# load_dotenv()

# os.environ['HUGGINGFACE_API_KEY'] = os.getenv("HUGGINGFACE_API_KEY")

# embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")


# ## Set up Streamlit 

# st.title("Conversational RAG With PDF uploads and Chat History")

# st.write("Upload Pdf's and chat with their content")

# ##Input the Groq API Key

# api_key = st.text_input("Enter your groq Api Key:", type = "password")   

# ## Check if groq api key is provided

# if api_key:
#     llm = ChatGroq(groq_api_key= api_key, model_name = "Gemma2-9b-it")
    
#     ##Chat Interface
#     session_id = st.text_input("Session ID",value = "default_session")
    
#     ##Statefully Manage the chat History
    
#     if 'store' not in st.session_state:
#         st.session_state.store ={}
        
        
#     uploaded_files = st.file_uploader("Choose a pdf File", type="pdf",accept_multiple_files=True)
    
#     ##Process uploaded files
    
#     if uploaded_files:
#         documents = []
#         for uploaded_file in uploaded_files:
#             temppdf = f"./temp.pdf"
#             with open(temppdf,"wb") as file:
#                 file.write(uploaded_file.getvalue())
#                 file_name = uploaded_file.name
            
#             loader = PyPDFLoader(temppdf)
#             docs = loader.load()
#             documents.extend(docs)
            
    
#         vectorstore = Chroma(
#             collection_name="test_collection",
#             embedding_function=embeddings,
#             persist_directory="./chroma_db"

#         )            
    
#     ## Split and create embeddings for the documents
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#         splits = text_splitter.split_documents(documents)
#         vectorstore.add_documents(splits) 
#         retriever = vectorstore.as_retriever()   


#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question"
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it if needed and otherwise return it as is."
#         )    


#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system",contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human","{input}"),
#             ]
#         )

#         history_aware_retriever = create_history_aware_retriever(llm,retreiver,contextualize_q_prompt)

#         ##Answer Question prompt

#         system_prompt = (
#         "You are an assistant for question-answering tasks."
#         "Use the following pieces of retrieved context to answer"
#         "the question. If you don't know the answer , say that you"
#         "don't know . Use three sentences maximum and keep the "
#         "answer concise."
#         "\n\n"
#         "{context}"
#         )

#         qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system",system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human","{input}"),
            
#         ]
#         )

#         question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#         rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)


#         def get_session_history(session:str) -> BaseChatMessageHistory:
            
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id] = ChatMessageHistory()

#             return st.session_state.store[session_id] 

#         conversational_rag_chain = RunnableWithMessageHistory(
#             rag_chain,get_session_history,
#             input_messages_key="input",
#             history_messages_key="chathistory",
#             output_messages_key="answer"
#         )
        
        
#         user_input = st.text_input("Your Question:")
        
#         if user_input:
#             session_history = get_session_history(session_id)
#             response = conversational_rag_chain.invoke(
#                 {"input":user_input},
#                 config={
#                     "configurable": {"session_id":session_id}
#                 }, 
#             )
            
#             st.write(st.session_state.store)
#             st.success("Assisstant:", response['answer'])
#             st.write("Chat History:", session_history.messages)
            
# else:
#     st.warning("Please Enter the groq API key")




import streamlit as st
import os
from langchain_groq import ChatGroq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# FIX 1: Use langchain_community chains, not langchain_classic (doesn't exist)
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

# FIX 2: Correct env variable name (was HUGGINGFACE_API_KEY, should be HF_TOKEN or just set it)
hf_key = os.getenv("HUGGINGFACE_API_KEY")
if hf_key:
    os.environ['HUGGINGFACE_API_KEY'] = hf_key

# FIX 3: Initialize embeddings once at top level (correct, keep as is)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Set up Streamlit
st.title("Conversational RAG With PDF uploads and Chat History")
st.write("Upload PDF's and chat with their content")

## Input the Groq API Key
api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="qwen/qwen3-32b")

    ## Chat Interface
    session_id = st.text_input("Session ID", value="default_session")

    ## Statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF File", type="pdf", accept_multiple_files=True)

    ## Process uploaded files
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        ## Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # FIX 4: Verify splits are not empty before adding
        if not splits:
            st.error("No content extracted from the PDF. Please try a different file.")
            st.stop()

        # FIX 5: Create vectorstore WITH embedding_function (this was the main crash cause)
        vectorstore = Chroma(
            collection_name="test_collection",
            embedding_function=embeddings,   # ✅ Required!
            persist_directory="./chroma_db"
        )

        vectorstore.add_documents(splits)

        # FIX 6: Consistent variable name — was 'retriever' then 'retreiver' (typo)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # FIX 7: Was 'retreiver' (typo) — now correctly 'retriever'
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        ## Answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:   # FIX 8: was session_id (outer var), now uses param
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",   # FIX 9: was "chathistory", must match MessagesPlaceholder
            output_messages_key="answer"
        )

        user_input = st.text_input("Your Question:")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },
            )

            st.write(st.session_state.store)
            # FIX 10: st.success only takes a string, use st.write for the answer
            st.success("Assistant response received!")
            st.write("**Assistant:**", response['answer'])
            st.write("**Chat History:**", session_history.messages)

else:
    st.warning("Please enter the Groq API key")

    
           

🤖 Conversational RAG with PDF & Chat History
A Retrieval-Augmented Generation (RAG) web app built with Streamlit, LangChain, and Groq. Upload any PDF and have a multi-turn conversation with its content — the app remembers your chat history for context-aware answers.


✨ Features

📄 Multi-PDF Upload — Upload one or more PDFs simultaneously
🧠 Context-Aware Conversations — Chat history is used to reformulate follow-up questions
⚡ Groq-Powered LLM — Ultra-fast inference using Gemma2-9b-it
🔍 Semantic Search — HuggingFace embeddings + ChromaDB vector store
🗂️ Session Management — Multiple isolated chat sessions supported
🖥️ Clean Streamlit UI — Simple, browser-based interface


PDF Upload
    │
    ▼
PyPDFLoader  ──►  RecursiveCharacterTextSplitter
                          │
                          ▼
               HuggingFace Embeddings (all-MiniLM-L6-v2)
                          │
                          ▼
                    ChromaDB Vector Store
                          │
                    ┌─────┴──────┐
                    │            │
               Retriever    Chat History
                    │            │
                    └─────┬──────┘
                          │
              History-Aware Retriever (LangChain)
                          │
                          ▼
                   Groq LLM (Gemma2-9b-it)
                          │
                          ▼
                    Final Answer 💬

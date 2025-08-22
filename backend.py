import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from supabase.client import Client, create_client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

# ------------------------
# 1. Load environment
# ------------------------
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# ------------------------
# 2. Initialize Supabase
# ------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------
# 3. Initialize embeddings & vector store
# ------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# ------------------------
# 4. Initialize LLM
# ------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ------------------------
# 5. FastAPI app
# ------------------------
app = FastAPI(title="Oral Health RAG Chatbot Backend")

# Allow your frontend origin
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://huggingface.co/spaces/Dylan4353847/chompbot"
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# 6. In-memory conversation storage
# ------------------------
conversations: Dict[str, List[str]] = {}

# ------------------------
# 7. Request model
# ------------------------
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"  # Simple session identifier

# ------------------------
# 8. Chat endpoint
# ------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    """
    Endpoint to handle multi-turn chat requests.
    """
    # Get or create conversation history for this session
    if req.session_id not in conversations:
        conversations[req.session_id] = []
    
    session_history = conversations[req.session_id]
    
    # Build chat history for context
    chat_history = []
    for i, msg in enumerate(session_history):
        if i % 2 == 0:
            chat_history.append(HumanMessage(msg))
        else:
            chat_history.append(AIMessage(msg))

    # Retrieve relevant documents from Supabase
    retrieved_docs = vector_store.similarity_search(req.question, k=2)
    context = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )

    # Optional: print context for debugging


    # Build prompt including conversation history and retrieved context
    conversation = ""
    for i, msg in enumerate(chat_history):
        conversation += f"{msg.content}\n"

    prompt = f"""
You are a friendly oral health chatbot. Your goal is to help users solve oral health issues. 

- Before giving any solution, always ask 1–2 clarifying questions to understand the user's situation fully.
- Use information from past conversation messages and relevant documents to guide your questions.
- Once you have enough information, provide a clear and practical solution.
- Always respond in a helpful, empathetic, and friendly manner.


Conversation so far:
{conversation}

Relevant documents:
{context}

Next question: {req.question}
"""

    # Call LLM using invoke()
    response = llm.invoke([HumanMessage(content=prompt)])

    # Store this conversation turn in memory
    conversations[req.session_id].append(req.question)
    conversations[req.session_id].append(response.content)

    # Return the answer
    return {"answer": response.content}

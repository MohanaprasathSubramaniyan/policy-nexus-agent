import os
import chainlit as cl
import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.tools.tavily_research import TavilyToolSpec

# 1. SETUP
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 2. CONFIGURATION
QDRANT_PATH = "./qdrant_data"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"

# --- NEW: UI BUTTONS (Option A) ---
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Internal Policy",
            message="What are the reporting requirements for telework?",
            icon="/public/learn.svg",
            ),
        cl.Starter(
            label="Live Web Search",
            message="What is the current stock price of Apple?",
            icon="/public/terminal.svg",
            ),
        cl.Starter(
            label="General Chat",
            message="Hello! Who are you?",
            icon="/public/idea.svg",
            )
        ]

@cl.on_chat_start
async def start():
    print("🚀 Starting PolicyNexus (v2.0)...")

    # -- Check Keys --
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        await cl.Message(content="❌ **Error:** API Keys missing in .env").send()
        return

    # 3. INITIALIZE MODELS
    try:
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        Settings.llm = Groq(model=LLM_MODEL, api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        await cl.Message(content=f"❌ Model Error: {str(e)}").send()
        return

    # 4. PREPARE TOOLS
    try:
        # Tool A: PDF Engine
        if os.path.exists(QDRANT_PATH):
            client = qdrant_client.QdrantClient(path=QDRANT_PATH)
            vector_store = QdrantVectorStore(client=client, collection_name="policy_docs")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            pdf_engine = index.as_query_engine(similarity_top_k=3)
        else:
            pdf_engine = None

        # Tool B: Web Search
        tavily_tool = TavilyToolSpec(api_key=os.environ.get("TAVILY_API_KEY"))

    except Exception as e:
        await cl.Message(content=f"❌ Tool Error: {str(e)}").send()
        return

    # Save to session
    cl.user_session.set("pdf_engine", pdf_engine)
    cl.user_session.set("tavily_tool", tavily_tool)
    
    await cl.Message(content="🛡️ **Nexus Online:** I am ready. Select a starter question below!").send()


@cl.on_message
async def main(message: cl.Message):
    pdf_engine = cl.user_session.get("pdf_engine")
    tavily_tool = cl.user_session.get("tavily_tool")
    llm = Settings.llm

    # --- ROUTER LOGIC (Option B: Smarter Greeting Handling) ---
    # We added a third option: "CHAT" for simple greetings
    router_prompt = (
        "You are a routing system. "
        "1. If the query is about specific Internal Policy, rules, or work hours, return 'INTERNAL'. "
        "2. If the query requires outside knowledge, news, or stocks, return 'EXTERNAL'. "
        "3. If the query is just a greeting like 'hi', 'hello', or 'who are you', return 'CHAT'. "
        f"Classify this query: '{message.content}'. "
        "Reply ONLY with one word: INTERNAL, EXTERNAL, or CHAT."
    )
    
    classification = str(llm.complete(router_prompt)).strip().upper()
    print(f"🤖 Decision: {classification}")

    response_text = ""
    
    # --- PATH 1: INTERNAL PDF ---
    if "INTERNAL" in classification and pdf_engine:
        status_msg = cl.Message(content="📄 *Consulting policy documents...*")
        await status_msg.send()
        response = pdf_engine.query(message.content)
        response_text = str(response)
        await status_msg.remove()

    # --- PATH 2: GENERAL CHAT (No Tools) ---
    elif "CHAT" in classification:
        # Just answer directly without searching!
        response_text = llm.complete(f"Reply politely to this user message: {message.content}")

    # --- PATH 3: WEB SEARCH (Fallback) ---
    else:
        status_msg = cl.Message(content="🌍 *Searching the global web...*")
        await status_msg.send()
        
        search_results = tavily_tool.search(message.content, max_results=3)
        summary_prompt = (
            f"User Question: {message.content}\n\n"
            f"Search Results: {search_results}\n\n"
            "Answer the question using the search results. Cite sources."
        )
        response_text = llm.complete(summary_prompt)
        await status_msg.remove()

    await cl.Message(content=str(response_text)).send()
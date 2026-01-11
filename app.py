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

@cl.on_chat_start
async def start():
    print("🚀 Starting Manual Router Bot...")

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

    # 4. PREPARE TOOLS (But don't put them in an Agent)
    try:
        # --- Tool A: PDF Engine ---
        if os.path.exists(QDRANT_PATH):
            client = qdrant_client.QdrantClient(path=QDRANT_PATH)
            vector_store = QdrantVectorStore(client=client, collection_name="policy_docs")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            
            # Create the query engine (we will call this manually)
            pdf_engine = index.as_query_engine(similarity_top_k=3)
        else:
            pdf_engine = None

        # --- Tool B: Web Search ---
        tavily_tool = TavilyToolSpec(api_key=os.environ.get("TAVILY_API_KEY"))

    except Exception as e:
        await cl.Message(content=f"❌ Tool Error: {str(e)}").send()
        return

    # Save tools to session
    cl.user_session.set("pdf_engine", pdf_engine)
    cl.user_session.set("tavily_tool", tavily_tool)
    
    await cl.Message(content="🌐 **System Online:** I am ready. I will decide whether to check the **PDF** or the **Web** based on your question.").send()


@cl.on_message
async def main(message: cl.Message):
    # Retrieve tools
    pdf_engine = cl.user_session.get("pdf_engine")
    tavily_tool = cl.user_session.get("tavily_tool")
    llm = Settings.llm

    # --- STEP 1: THE ROUTER ---
    # We ask the LLM to classify the intent
    router_prompt = (
        "You are a routing system. specific Internal Telework Policy rules, agreements, hours of work, or reporting belong to 'INTERNAL'. "
        "General questions, news, stocks, weather, or outside knowledge belong to 'EXTERNAL'. "
        f"Classify this query: '{message.content}'. "
        "Reply ONLY with the word 'INTERNAL' or 'EXTERNAL'."
    )
    
    classification = str(llm.complete(router_prompt)).strip().upper()
    print(f"🤖 Router Decision: {classification}")

    # --- STEP 2: EXECUTE ---
    response_text = ""
    
    if "INTERNAL" in classification and pdf_engine:
        # Use PDF
        status_msg = cl.Message(content="📄 *Checking internal policy documents...*")
        await status_msg.send()
        
        response = pdf_engine.query(message.content)
        response_text = str(response)
        
        await status_msg.remove() # Clean up status message

    else:
        # Use Web (Default)
        status_msg = cl.Message(content="🌍 *Searching the web...*")
        await status_msg.send()
        
        # Tavily search returns a list of results, we need to ask the LLM to summarize them
        search_results = tavily_tool.search(message.content, max_results=3)
        
        # Summarize the search results
        summary_prompt = (
            f"User Question: {message.content}\n\n"
            f"Search Results: {search_results}\n\n"
            "Please answer the user's question using the search results above. Cite your sources."
        )
        response_text = llm.complete(summary_prompt)
        
        await status_msg.remove()

    # --- STEP 3: REPLY ---
    await cl.Message(content=str(response_text)).send()
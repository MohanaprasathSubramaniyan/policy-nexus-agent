import os
import qdrant_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuration
LOCAL_FILE = "temp_policy.pdf"
QDRANT_PATH = "./qdrant_data"

def ingest():
    # 1. Verify the file exists locally
    if not os.path.exists(LOCAL_FILE):
        print(f"❌ ERROR: I cannot find '{LOCAL_FILE}'")
        print("👉 Please download the PDF manually, rename it to 'temp_policy.pdf', and drag it into this folder.")
        return

    # 2. Configure Embeddings
    print("🧠 Initializing Embedding Model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 3. Load Data
    print("📄 Loading documents...")
    documents = SimpleDirectoryReader(input_files=[LOCAL_FILE]).load_data()

    # 4. Initialize Qdrant Client
    print(f"💾 Creating Database in '{QDRANT_PATH}'...")
    client = qdrant_client.QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name="policy_docs")
    
    # 5. Create Storage Context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 6. Index Data
    print("⚙️ Indexing data (this may take a moment)...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print(f"🎉 Success! Database saved to '{QDRANT_PATH}'")

if __name__ == "__main__":
    ingest()
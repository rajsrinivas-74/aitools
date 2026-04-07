"""
Demo script: Vector Search with PDF Document
Shows how to load documents and perform semantic searches.
"""

import asyncio
import json
from pathlib import Path
from vector_search_agent_tool import VectorSearchAgentTool, create_basic_vector_search_tool
from simple_vector_store import SimpleVectorStore
from simple_embedding_provider import SimpleEmbeddingProvider


async def main():
    """Main demo function."""
    
    print("\n" + "="*70)
    print("🚀 VECTOR SEARCH TOOL - PDF DOCUMENT DEMO")
    print("="*70 + "\n")
    
    # Step 1: Create and configure the tool
    print("📋 Step 1: Initializing Vector Search Tool...")
    tool = create_basic_vector_search_tool(config={
        "index_path": "faiss_index",
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_dim": 384,
    })
    
    # Configure the backends
    vector_store = SimpleVectorStore(save_path="pdf_vector_index.json")
    embedding_provider = SimpleEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    
    tool.set_vector_store(vector_store)
    tool.set_embedding_provider(embedding_provider)
    
    print("✓ Vector store configured")
    print("✓ Embedding provider configured\n")
    
    # Step 2: Load documents from the documents folder
    print("📂 Step 2: Loading documents from 'documents' folder...")
    result = await tool.load_documents_from_folder('documents')
    result_data = json.loads(result)
    
    print(json.dumps(result_data, indent=2))
    print()
    
    if result_data.get("loaded", 0) == 0:
        print("❌ No documents loaded. Please check the documents folder.")
        return
    
    # Step 3: Show tool statistics
    print("📊 Step 3: Tool Statistics")
    stats = tool.get_stats()
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Vector store configured: {stats['vector_store_configured']}")
    print(f"   Embedding provider configured: {stats['embedding_provider_configured']}")
    print()
    
    # Step 4: Perform searches
    print("🔍 Step 4: Performing Semantic Searches\n")
    
    search_queries = [
        "What is AI orchestration?",
        "How does workflow automation work?",
        "Tell me about intelligent agents",
        "What are the key benefits of AI?",
        "How to implement AI workflows?"
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"Query {i}: '{query}'")
        print("-" * 70)
        
        search_result = await tool.search_knowledge_base(query, top_k=2)
        results = json.loads(search_result)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"Found: {results['count']} results\n")
            
            for j, result in enumerate(results['results'], 1):
                score = result['score']
                content = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
                print(f"  Result {j} (score: {score:.3f}):")
                print(f"    {content}\n")
        
        print()
    
    # Step 5: Show what's been saved
    print("💾 Step 5: Data Persistence")
    index_file = Path("pdf_vector_index.json")
    if index_file.exists():
        size_kb = index_file.stat().st_size / 1024
        print(f"✓ Vector index saved to: {index_file}")
        print(f"  File size: {size_kb:.1f} KB")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETED")
    print("="*70)
    print("\nYou can now use the vector search tool to:")
    print("  1. Load documents from the documents folder")
    print("  2. Search using semantic queries")
    print("  3. Get relevant results based on vector similarity")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

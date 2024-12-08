import gc
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time

faiss_index_dir = "../embeddings/faiss_index"


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore = FAISS.load_local(faiss_index_dir, embedding_model, allow_dangerous_deserialization=True)
print("Loaded FAISS index from disk.")

user_query = input("Enter your question: ")


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3, "fetch_k": 4})
start_time_query = time.time()
results = retriever.get_relevant_documents(user_query)
end_time_query = time.time()

print(f"\nQuery processed in {end_time_query - start_time_query:.4f} seconds.")

if results:
    print("\nRetrieved Results:")
    for idx, result in enumerate(results, start=1):
        print(f"\nResult {idx}:")
        print(f"Context: {result.page_content}")
        print(f"Title: {result.metadata.get('title', 'Unknown Title')}")
        print(f"URL: {result.metadata.get('url', 'No URL')}")
        print(f"Topic: {result.metadata.get('topic', 'Unknown Topic')}")
else:
    print("\nNo relevant context found.")


del results, retriever
gc.collect()

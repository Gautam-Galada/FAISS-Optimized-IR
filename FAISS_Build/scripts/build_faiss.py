import pickle
import time
import gc
from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

input_pickle = "../data/split_documents.pkl"
faiss_index_dir = "../embeddings/faiss_index"

with open(input_pickle, 'rb') as f:split_documents = pickle.load(f)

print(f"Loaded {len(split_documents)} document chunks for embedding.")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

batch_size = 50000
print("Creating embeddings...")
start_time = time.time()

faiss_index = None
for batch_num, start_idx in enumerate(tqdm(range(0, len(split_documents), batch_size), desc="Processing embeddings")):
    end_idx = min(start_idx + batch_size, len(split_documents))
    batch_docs = split_documents[start_idx:end_idx]
    batch_texts = [doc.page_content for doc in batch_docs]
    batch_metadatas = [doc.metadata for doc in batch_docs]
    if faiss_index is None:faiss_index = FAISS.from_texts(texts=batch_texts,embedding=embedding_model,metadatas=batch_metadatas)
    else:faiss_index.add_texts(batch_texts, batch_metadatas)
    del batch_docs,batch_texts,batch_metadatas
    gc.collect()

faiss_index.save_local(faiss_index_dir)

end_time = time.time()
print(f"Embeddings created and saved to FAISS index in {end_time - start_time:.2f} seconds.")

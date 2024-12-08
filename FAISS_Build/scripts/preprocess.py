import json
import pickle
import gc
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


json_file = "../data/cleaned_wiki_articles.json"
output_pickle = "../data/split_documents.pkl"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

with open(json_file, 'r') as f:data = json.load(f)

split_documents = []

print("Processing and chunking documents...")
for topic, articles in tqdm(data.items(), desc="Processing topics"):
    for article in tqdm(articles, desc=f"Processing topic: {topic}"):
        title = article.get("title", "Unknown Title")
        url = article.get("url", "No URL")
        summary = article.get("summary", "")
        metadata = {"topic": topic,"title": title, "url": url,}
        chunks = text_splitter.split_text(summary)
        split_documents.extend([
            Document(
                page_content=chunk,
                metadata=metadata
            ) for chunk in chunks
        ])

    gc.collect()


with open(output_pickle, 'wb') as f:pickle.dump(split_documents, f)

print(f"Total chunks created: {len(split_documents)}")
print(f"Chunked documents saved to: {output_pickle}")

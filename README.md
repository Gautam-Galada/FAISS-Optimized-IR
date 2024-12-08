# **Wikipedia-Based Information Retrieval System**

## **Introduction**

In modern computational systems, efficient Information Retrieval (IR) is pivotal for large-scale data analysis, especially when spanning multiple domains. This project, developed by **Team Aviato**, presents an advanced IR system that integrates Natural Language Processing (NLP), FAISS indexing, and conversational AI using Ollama’s LLM (**Llama 3**). By combining cutting-edge technologies, the system retrieves relevant information from Wikipedia and handles user queries through an intuitive **Flask-based interface**.

The motivation for using **FAISS** (Facebook AI Similarity Search) lies in its ability to perform fast, scalable, and approximate nearest neighbor searches, which are essential for embedding-based retrieval systems. The inclusion of Llama 3 ensures contextually precise responses for both conversational (chitchat) and information-seeking queries.

### **Key Statistics**
- **Total Articles Scraped**: 50,000+  
- **Average Query Response Time**: ~8 seconds  

---

## **Methodology**

<div align="center">
    <img src="https://github.com/user-attachments/assets/7333dccf-4629-4570-834f-188e1d8d525f" alt="aviato" />
</div>


### **1. Scraping Wikipedia**
The system starts with data acquisition by scraping over **5000 articles per topic** from Wikipedia across domains like **Health**, **Environment**, **Technology**, **Economy**, and others. Subtopics were identified to ensure comprehensive coverage, making the scraping process robust and domain-specific.

### **2. Data Cleaning**
The scraped data undergoes rigorous preprocessing to ensure quality:
- **HTML Parsing**: Removed HTML tags and entities using BeautifulSoup.  
- **Text Normalization**: Standardized punctuation and whitespace.  
- **Noise Reduction**: Removed non-alphanumeric characters and references.  
- **Semantic Filtering**: Lowercased text for compatibility with embedding models.

### **3. NLP Techniques for Embedding**
Embeddings are generated using **HuggingFace’s sentence-transformers**, specifically the `"paraphrase-MiniLM-L6-v2"` model. Key features of this model include:
- **Dimensionality Reduction**: Reduces computational overhead while retaining expressivity.
- **Semantic Clustering**: Captures contextual relationships for nearest neighbor searches.

### **4. Building FAISS Index**
The processed data is chunked into smaller segments (200 characters with a 20-character overlap). These chunks are embedded into dense vector spaces and indexed using **FAISS**:
- **Why FAISS?**
  - Optimized for large-scale vector search and nearest neighbor retrieval.
  - GPU acceleration ensures fast indexing and retrieval.
  - Metadata storage supports topic, title, and URL integration.

The **Maximum Marginal Relevance (MMR)** search algorithm is employed to balance relevance and diversity, ensuring comprehensive and non-redundant results.

### **5. Flask and Ollama Integration**
The Flask-based interface integrates FAISS for retrieval and Ollama’s Llama 3 for response generation. The workflow is as follows:
1. **Query Classification**: Identifies queries as chitchat or non-chitchat.  
2. **Document Retrieval**: FAISS retrieves the top relevant chunks using MMR.  
3. **Contextual Answering**: Llama 3 generates concise and specific responses.  
4. **Analytics Dashboard**: Tracks query statistics and topic distribution.

---

## **Hardware and Performance**

- **GPU**: NVIDIA A100 (40GB)  
- **System RAM**: 83.5GB  
- **Time Efficiency**:
  - FAISS Index Creation: ~22.4 minutes (1344 seconds)  
  - Average Query Response Time: ~8 seconds  


---
## Contributors

- [Gautam Galada](https://github.com/Gautam-Galada)
- [Anany Singh](https://github.com/Anany25)
- [Nikunj Sujit](https://github.com/nikunjsuj13)


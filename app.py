#PROJECT 3
#Anany Singh,Gautam Galada and Nikunj Odayoth

from flask import Flask, request, jsonify, render_template
from langchain_ollama import OllamaLLM
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import time
import json
from collections import defaultdict

app = Flask(__name__)
db_folder = "faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore = FAISS.load_local(db_folder, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1, "fetch_k": 1})
llm = OllamaLLM(model="llama3")


query_count = 0
chitchat_count = 0
non_chitchat_count = 0
topic_counts = defaultdict(int)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global query_count, chitchat_count, non_chitchat_count
    user_input = request.form.get("user_input")
    start_time = time.time() 
    query_count += 1

   
    result = llm.invoke(
    "Check if the text query is general chitchat or not "
    "If the query is a greeting or general chitchat, respond naturally and vary the responses to avoid repetition. Use conversational and contextually appropriate phrases. For example, responses to 'hi' could include: "
    "'Hello! How can I help you today?', 'Hi there! What can I do for you?', 'Hey! What's on your mind today?', or 'Hi! How's your day going?'. "
    "Another example: for 'good morning', possible responses are: 'Good morning! How can I assist you?', 'Morning! What would you like to know today?', or 'Good morning! How's your day so far?'. "
    "Another example: for 'how are you?' possible responses are 'I am fine! How are you?'"
    "Ensure that each greeting query gets a unique response from the available options. "
    "If the text is not a greeting, just reply with 'Not Greeting'"
    "Even text which have parts of chitchat but are not actually fully chitchat are clasified as 'Not Greeting'. For example- the query 'Hi what is yellow fever?' or 'Hey where is Lake Ontario?' or 'Hello when was World War 2?' is not a greeting as it is mainly a question and not general chitchat or a greeting."
    "Any question not directed directky at the chatbot should be classified as 'Not greeting'"
    "The text query is '" + user_input + "'."
    )

    if 'Not Greeting' in result:
        non_chitchat_count += 1
        user_query = user_input
        results = retriever.get_relevant_documents(user_query)

        results_list = [result.page_content for result in results]
        for result in results:
            matching_topic=result.metadata.get('topic', 'Unknown Topic')
        if not results_list:
            print("\nNo relevant documents found.")
        else:
            print(f"\nRetrieved {len(results_list)} Substrings:")
            for i, content in enumerate(results_list, start=1):
                print(f"Substring {i}: {content}")
                continue
        with open("articles.json", "r") as file:
            data = json.load(file)

        topic_counts[matching_topic] += 1

        
        if results_list:
            explicit_prompt_template = """
            You are an AI model that provides detailed and specific answers. Answer the question based on the context below. If you cannot answer the question, reply "I don't know". Limit your response to 30 words.

            Context: {context}

            Question: {question}
            """
            explicit_prompt = PromptTemplate.from_template(explicit_prompt_template)
            input_data = {"context": results_list[0], "question": user_query}

           
            answer = llm.invoke(input=explicit_prompt.format(**input_data))
        else:
            answer = "I don't know."
        end_time = time.time()  
        response_time = end_time - start_time  
        return jsonify({"response": answer, "category": "Non-Chitchat", "topic": matching_topic})
    else:
        chitchat_count += 1
        end_time = time.time()  
        response_time = end_time - start_time  
        return jsonify({"response": result, "category": "Chitchat"})


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "query_count": query_count,
        "chitchat_count": chitchat_count,
        "non_chitchat_count": non_chitchat_count,
        "topic_counts": topic_counts
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)

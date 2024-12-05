from flask import Flask, request, jsonify, render_template
from langchain_ollama import OllamaLLM
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json
from langchain.prompts import PromptTemplate


app = Flask(__name__)
db_folder = "faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore = FAISS.load_local(db_folder, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1, "fetch_k": 1})
llm = OllamaLLM(model="llama3")



# Initialize counters
chitchat_count = 0
non_chitchat_count = 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global chitchat_count, non_chitchat_count
    user_input = request.form.get("user_input")

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
        user_query=user_input
        
        results = retriever.get_relevant_documents(user_query)

        # Collect the substrings (retrieved documents)
        results_list = [result.page_content for result in results]

        if not results_list:
            print("\nNo relevant documents found.")
        else:
            print(f"\nRetrieved {len(results_list)} Substrings:")
            for i, content in enumerate(results_list, start=1):
                print(f"Substring {i}: {content}")
                continue

        with open("articles.json", "r") as file:
            data = json.load(file)

        # Define a function to find the topic for any matching substring
        def find_topic_for_any_substring(substrings, json_data):
            """
            Finds the topic for any matching substring by checking JSON summaries.
            
            Args:
                substrings (list): List of substrings (retrieved document contents).
                json_data (dict): The JSON data containing topics and their documents.
            
            Returns:
                dict: The matching topic and associated metadata for the first matching substring,
                    or None if no match is found.
            """
            for substring in substrings:
                substring_lower = substring.lower()  # Case-insensitive comparison
                for topic, documents in json_data.items():
                    for doc in documents:
                        summary = doc.get("summary", "").lower()
                        if substring_lower in summary:  # Check if substring matches summary
                            return {
                                "topic": topic,
                                "title": doc.get("title", "Unknown Title"),
                                "url": doc.get("url", "Unknown URL")
                            }
            return None

        # Find the topic for any of the retrieved substrings
        matching_data = find_topic_for_any_substring(results_list, data)

        # Output the matching topic and metadata
        if matching_data:
            print(f"\nMatching Topic: {matching_data['topic']}")
            print(f"Title: {matching_data['title']}")
            print(f"URL: {matching_data['url']}")
        else:
            print("\nNo matching topic found for the retrieved substrings.")
        explicit_prompt_template = """
        You are an AI model that provides detailed and specific answers. Answer the question based on the context below. If you cannot answer the question, reply "I don't know". Limit your response to 30 words.

        Context: {context}

        Question: {question}
        """
        explicit_prompt = PromptTemplate.from_template(explicit_prompt_template)

        input_data = {"context": results_list[0], "question": user_query}

        #start_time = time.time()
        answer = llm.invoke(input=explicit_prompt.format(**input_data))
        #end_time = time.time()
        
        return jsonify({"response": answer, "category": "Non-Chitchat"})
    else:
        chitchat_count += 1
        return jsonify({"response": result, "category": "Chitchat"})


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "chitchat_count": chitchat_count,
        "non_chitchat_count": non_chitchat_count
    })


if __name__ == "__main__":
    app.run(debug=True)

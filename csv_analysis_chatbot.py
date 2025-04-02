import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import faiss
import sqlite3
from dotenv import load_dotenv

# Configure API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Database setup
conn = sqlite3.connect("csv_analysis_chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        bot_response TEXT
    )
""")
conn.commit()

### 1️⃣ Extract Text from CSV ###
def get_csv_text(csv_files):
    text = ""
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)  # Read CSV file
        text += df.to_string(index=False) + "\n"  # Convert to string format
    return text

### 2️⃣ Split Text into Chunks ###
def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks

### 3️⃣ Convert Chunks into Embeddings ###
def get_embeddings(text_chunks):
    model = "models/text-embedding-004"  # Best embedding model
    embeddings = []
    
    for chunk in text_chunks:
        response = genai.embed_content(model=model, content=chunk, task_type="retrieval_document")
        embeddings.append(response["embedding"])
    
    return np.array(embeddings, dtype=np.float32)

### 4️⃣ Store Embeddings in FAISS ###
def store_embeddings_in_faiss(embeddings):
    d = embeddings.shape[1]  
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

### 5️⃣ Search FAISS for Relevant Chunks ###
def retrieve_relevant_chunks(query, text_chunks, index):
    model = "models/text-embedding-004"  
    query_embedding = genai.embed_content(model=model, content=query, task_type="retrieval_query")["embedding"]
    
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    _, closest_indices = index.search(query_vector, k=4)
    
    return " ".join([text_chunks[i] for i in closest_indices[0]])

### 6️⃣ Generate AI Response Using Gemini ###
def generate_response(context, user_question):
    model = "models/gemini-1.5-pro-latest"  

    if not context:
        return "I need the CSV content to answer your question. Please upload a CSV file first."

    prompt = f"""
    Answer the question based on the provided context. If the answer is not available in the context, say "The sole information present in the CSV is inconclusive for the asked question. More data and context is needed. I apologise, I am currently under development!"
    
    Context: {context}
    Question: {user_question}

    Answer:
    """

    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(prompt)

    return response.text

# Store chat messages and CSV content
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "csv_content" not in st.session_state:
    st.session_state["csv_content"] = ""

### 7️⃣ Main Streamlit App ###
def main():
    st.set_page_config("Chat CSV", layout="wide")  
    st.title("CSV Analysis Chatbot")

    # Load chat history from database
    cursor.execute("SELECT user_input, bot_response FROM chat")
    chats = cursor.fetchall()

    # Custom CSS for formatting
    st.markdown("""
        <style>
            .bot-message {
                text-align: left;
                background-color: #f1f1f1;
                padding: 10px;
                border-radius: 10px;
                width: 60%;
                margin-bottom: 20px;
            }
            .user-message {
                text-align: left;
                background-color: #f1f1f1;
                color: black;
                padding: 10px;
                border-radius: 10px;
                width: 60%;
                margin-left: 40%;
                margin-bottom: 20px;
            }
            .chat-container {
                display: flex;
                flex-direction: column;
            }
            .chat-input textarea {
            width: 60% !important;
            margin-left: 40% !important;
            height: 50px !important;
            text-align: left !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display previous chat history
    for user_input, bot_response in chats:
        st.markdown(f'<div class="user-message">🧑 {user_input}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-message">🤖 {bot_response}</div>', unsafe_allow_html=True)

    # Multi-line text input aligned properly
    user_question = st.text_area("Type your message here:", key="user_input")

    if st.button("Send", use_container_width=True):
        if user_question:
            st.session_state["messages"].append({"role": "user", "content": user_question})

            # Generate bot response using stored CSV text
            response = generate_response(st.session_state["csv_content"], user_question)

            # Store in session state
            st.session_state["messages"].append({"role": "bot", "content": response})

            # Store in database
            cursor.execute("INSERT INTO chat (user_input, bot_response) VALUES (?, ?)", (user_question, response))
            conn.commit()

            # Display the chat
            st.markdown(f'<div class="user-message">🧑 {user_question}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">🤖 {response}</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("Menu:")
        csv_files = st.file_uploader("Upload your CSV Files", accept_multiple_files=True, type=["csv"])
        if st.button("Process CSV"):
            with st.spinner("Processing..."):
                raw_text = get_csv_text(csv_files)
                st.session_state["csv_content"] = raw_text  
                st.success("CSV uploaded and processed successfully!")

        if st.button("Clear Chat History"):
            cursor.execute("DELETE FROM chat")
            conn.commit()
            st.session_state["messages"] = []
            st.session_state["csv_content"] = ""
            st.rerun()


main()

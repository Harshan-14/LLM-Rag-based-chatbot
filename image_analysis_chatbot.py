
import streamlit as st
import os
import numpy as np
import faiss
import sqlite3
import google.generativeai as genai
import pytesseract
from PIL import Image
from dotenv import load_dotenv

# ---- Load API Key ----
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---- Database Setup ----
conn = sqlite3.connect("image_analysis_chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        bot_response TEXT
    )
""")
conn.commit()


def analyze_image_content(image, is_chart=False):
    """Analyze image content using Gemini Pro Vision"""
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        
        if is_chart:
            prompt = """
            Analyze this chart/graph in detail:
            1. Type of visualization
            2. Key trends and patterns
            3. Main insights and findings
            4. Data relationships shown
            5. Any notable outliers or interesting points
            
            Provide a comprehensive analysis that could help answer questions about this visualization.
            """
        else:
            prompt = """
            Analyze this image and extract:
            1. Any visible text
            2. If there are charts/graphs, describe their content
            3. Key information and insights
            4. Important numbers or data points
            
            Provide a detailed analysis that could help answer questions about this content.
            """

        # ✅ Pass the prompt and image inside a list
        response = model.generate_content([prompt, image])

        return response.text
    except Exception as e:
        st.error(f"Vision Analysis Error: {str(e)}")
        return ""




###1️⃣ Extract Text from Image Using OCR ###
def extract_text_from_images(images):
    """Extract both text and visual content from images"""
    extracted_text = ""
    visual_analysis = ""
    
    for image in images:
        img = Image.open(image).convert("RGB")
        
        # OCR for text
        text = pytesseract.image_to_string(img)
        extracted_text += text + "\n"
        
        # Visual analysis for charts/graphs
        visual_analysis += analyze_image_content(img) + "\n"
        
        # Specific chart analysis if detected
        if "graph" in text.lower() or "chart" in text.lower():
            chart_analysis = analyze_image_content(img, is_chart=True)
            visual_analysis += f"\nChart Analysis:\n{chart_analysis}\n"
    
    combined_analysis = f"""
    Extracted Text:
    {extracted_text}
    
    Visual Analysis:
    {visual_analysis}
    """
    
    return combined_analysis.strip()

### 2️⃣ Split Extracted Text into Chunks ###
def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks

### 3️⃣ Convert Chunks into Embeddings (Gemini) ###
def get_embeddings(text_chunks):
    model = "models/embedding-001"
    embeddings = []
    
    for chunk in text_chunks:
        response = genai.embed_content(
            model=model,
            content=chunk,
            task_type="retrieval_document"
        )
        embeddings.append(response["embedding"])
    
    return np.array(embeddings, dtype=np.float32)

### 4️⃣ Store Embeddings in FAISS ###
def store_embeddings_in_faiss(embeddings):
    d = len(embeddings[0])  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

### 5️⃣ Retrieve Relevant Chunks from FAISS ###
def retrieve_relevant_chunks(query, text_chunks, index):
    model = "models/embedding-001"
    query_embedding = genai.embed_content(
        model=model,
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    
    query_vector = np.array([query_embedding], dtype=np.float32)
    k = min(3, len(text_chunks))  # Get top 3 or less if fewer chunks exist
    _, indices = index.search(query_vector, k)
    
    relevant_chunks = [text_chunks[idx] for idx in indices[0]]
    return " ".join(relevant_chunks)

### 6️⃣ Generate Response Using Gemini ###
def generate_response(context, user_question):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")

        if not context:
            return "I need an image to analyze and answer your question. Please upload an image first."

        prompt = f"""
        Answer the question based on the provided context which includes both text and visual analysis. 
        If the question is about:
        - Charts/Graphs: Focus on the visual analysis section
        - Text content: Focus on the extracted text section
        - General understanding: Consider both sections
        
        If the answer is not available in the context, say:
        "I cannot find enough information in the image to answer this question. Please try another question or upload a different image."
        
        Context: {context}
        Question: {user_question}

        Answer:
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, there was an error generating the response. Please try again."
    

# ---- Store Chat Messages & Extracted Text ----
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "image_content" not in st.session_state:
    st.session_state["image_content"] = ""
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None
if "text_chunks" not in st.session_state:
    st.session_state["text_chunks"] = []

### 7️⃣ Streamlit App UI ###
def main():
    st.set_page_config("Chat with Image", layout="wide")  
    st.title("📷 Image-Based Q&A Chatbot")

    # Load chat history from database
    cursor.execute("SELECT user_input, bot_response FROM chat")
    chats = cursor.fetchall()

    # ---- Custom CSS ----
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

    # ---- Display Previous Chat History ----
    for user_input, bot_response in chats:
        st.markdown(f'<div class="user-message">🧑 {user_input}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-message">🤖 {bot_response}</div>', unsafe_allow_html=True)

    # ---- User Question Input ----
    user_question = st.text_area("Type your message here:", key="user_input")

    if st.button("Send", use_container_width=True):
        if user_question:
            st.session_state["messages"].append({"role": "user", "content": user_question})

            # Retrieve relevant context using FAISS
            if st.session_state["faiss_index"] is not None:
                relevant_context = retrieve_relevant_chunks(
                    user_question,
                    st.session_state["text_chunks"],
                    st.session_state["faiss_index"]
                )
            else:
                relevant_context = st.session_state["image_content"]

            # Generate response
            response = generate_response(relevant_context, user_question)

            # Store in session state
            st.session_state["messages"].append({"role": "bot", "content": response})

            # Store in database
            cursor.execute("INSERT INTO chat (user_input, bot_response) VALUES (?, ?)", 
                         (user_question, response))
            conn.commit()

            # Display the chat
            st.markdown(f'<div class="user-message">🧑 {user_question}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">🤖 {response}</div>', unsafe_allow_html=True)

    # ---- Sidebar for Image Upload ----
    with st.sidebar:
        st.title("📂 Upload Images")
        st.write("Support for both text documents and data visualizations")
        
        uploaded_images = st.file_uploader(
            "Upload your Images", 
            accept_multiple_files=True, 
            type=["png", "jpg", "jpeg"]
        )
        
        analysis_type = st.radio(
            "Content Type (helps improve analysis)",
            ["Auto Detect", "Mostly Text", "Charts/Graphs", "Mixed Content"]
        )
        
        if st.button("Process Image"):
            with st.spinner("Analyzing content..."):
                # Extract text and analyze visual content
                combined_analysis = extract_text_from_images(uploaded_images)
                st.session_state["image_content"] = combined_analysis

                # Split text into chunks
                text_chunks = split_text_into_chunks(combined_analysis)
                st.session_state["text_chunks"] = text_chunks

                # Get embeddings and store in FAISS
                if text_chunks:
                    embeddings = get_embeddings(text_chunks)
                    if embeddings is not None:
                        faiss_index = store_embeddings_in_faiss(embeddings)
                        st.session_state["faiss_index"] = faiss_index
                        st.success("Image processed successfully! Content analyzed and indexed.")
                    else:
                        st.warning("Error generating embeddings.")
                else:
                    st.warning("No content was extracted from the images.")


if __name__ == "__main__":
    main()
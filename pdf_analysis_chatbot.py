import streamlit as st
from PyPDF2 import PdfReader
import os
import google.generativeai as genai
import numpy as np
import faiss
import sqlite3
from dotenv import load_dotenv
from PIL import Image
import io


# Configure API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Database setup
conn = sqlite3.connect("pdf_analysis_chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        bot_response TEXT
    )
""")
conn.commit()

### 1️⃣ Extract Text and Images from PDF ###
def get_pdf_content(pdf_files):
    text_content = ""
    image_list = []
    
    for pdf_file in pdf_files:
        try:
            # Extract text using PyPDF2
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
                
                # Extract images using PyPDF2
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            try:
                                data = xObject[obj]._data
                                img = Image.open(io.BytesIO(data))
                                image_list.append(img)
                            except Exception as e:
                                st.warning(f"Failed to extract an image: {str(e)}")
                                continue
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            continue
    
    return text_content, image_list


### 2️⃣ Analyze Image Content ###
def analyze_image_content(image, is_chart=False):
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
            1. Key visual elements
            2. Any visible text
            3. Important details or features
            4. Context and relevance
            
            Provide a detailed analysis that could help answer questions about this content.
            """

        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        st.error(f"Vision Analysis Error: {str(e)}")
        return ""

### 3️⃣ Process All Content ###
def process_content(text_content, image_list):
    combined_content = text_content + "\n\n"
    
    # Analyze each image
    for idx, image in enumerate(image_list):
        st.sidebar.image(image, caption=f"Image {idx + 1}")
        
        # Check if image might be a chart
        is_chart = False  # You can implement chart detection logic here
        image_analysis = analyze_image_content(image, is_chart)
        
        combined_content += f"\nImage {idx + 1} Analysis:\n{image_analysis}\n\n"
    
    return combined_content

### 4️⃣ Split Text into Chunks ###
def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks

### 5️⃣ Convert Chunks into Embeddings ###
def get_embeddings(text_chunks):
    model = "models/text-embedding-004"
    embeddings = []
    
    for chunk in text_chunks:
        response = genai.embed_content(model=model, content=chunk, task_type="retrieval_document")
        embeddings.append(response["embedding"])
    
    return np.array(embeddings, dtype=np.float32)

### 6️⃣ Store Embeddings in FAISS ###
def store_embeddings_in_faiss(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

### 7️⃣ Search FAISS for Relevant Chunks ###
def retrieve_relevant_chunks(query, text_chunks, index):
    model = "models/text-embedding-004"
    query_embedding = genai.embed_content(model=model, content=query, task_type="retrieval_query")["embedding"]
    
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    _, closest_indices = index.search(query_vector, k=4)
    
    return " ".join([text_chunks[i] for i in closest_indices[0]])

### 8️⃣ Generate AI Response Using Gemini ###
def generate_response(context, user_question):
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    if not context:
        return "I need the document content to answer your question. Please upload a PDF file first."

    prompt = f"""
    Answer the question based on the provided context, which includes both text and image analysis. 
    If the question is about:
    - Images or visuals: Focus on the image analysis sections
    - Text content: Focus on the document text
    - General understanding: Consider both text and images
    
    If the answer is not available in the context, say:
    "I cannot find enough information in the document to answer this question completely. Please try another question or provide more context."
    
    Context: {context}
    Question: {user_question}

    Answer:
    """

    response = model.generate_content(prompt)
    return response.text

# Store chat messages and document content
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "doc_content" not in st.session_state:
    st.session_state["doc_content"] = ""
if "images" not in st.session_state:
    st.session_state["images"] = []

### 9️⃣ Main Streamlit App ###
def main():
    st.set_page_config("PDF Analysis", layout="wide")
    st.title("📄 PDF Analysis Chatbot")
    st.subheader("Upload PDF files with text and images")

    # Load chat history from database
    cursor.execute("SELECT user_input, bot_response FROM chat")
    chats = cursor.fetchall()

    # Custom CSS for a professional UI
    st.markdown("""
        <style>
            /* Main container styling */
            .main {
                padding: 2rem;
                padding-bottom: 200px;
            }
            
            /* Header styling */
            .stTitle {
                color: #1E88E5;
                font-size: 2.5rem !important;
                font-weight: 600;
                margin-bottom: 2rem;
            }
            
            /* Card styling */
            .css-1r6slb0 {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            /* Button styling */
            .stButton > button {
                width: 100%;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                background-color: #1E88E5;
                color: white;
                border: none;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                background-color: #1565C0;
                transform: translateY(-2px);
            }
            
            /* Chat container styling */
            .chat-container {
                background-color: white;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 2rem;
                margin-bottom: 100px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            
            /* User message styling */
            .user-message {
                background-color: #E3F2FD;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                width: 60%;
                margin-left: 40%;
                margin-bottom: 20px;
            }
            
            /* Bot message styling */
            .bot-message {
                background-color: #F5F5F5;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                width: 60%;
                margin-bottom: 20px;
            }

            /* Input container styling */
            .input-container {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background-color: white;
                padding: 20px;
                box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
                z-index: 100;
            }

            /* Sidebar styling */
            .sidebar .sidebar-content {
                margin-bottom: 100px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for document upload
    with st.sidebar:
        st.title("📂 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files (with images)",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    try:
                        # Extract text and images
                        text_content, image_list = get_pdf_content(uploaded_files)
                        
                        # Store images in session state
                        st.session_state["images"] = image_list
                        
                        # Process all content
                        combined_content = process_content(text_content, image_list)
                        st.session_state["doc_content"] = combined_content

                        # Split text into chunks
                        text_chunks = split_text_into_chunks(combined_content)
                        st.session_state["text_chunks"] = text_chunks

                        # Get embeddings and store in FAISS
                        if text_chunks:
                            embeddings = get_embeddings(text_chunks)
                            if embeddings is not None and len(embeddings) > 0:
                                faiss_index = store_embeddings_in_faiss(embeddings)
                                st.session_state["faiss_index"] = faiss_index
                                st.success("Documents processed successfully!")
                            else:
                                st.warning("Error generating embeddings.")
                        else:
                            st.warning("No content was extracted from the documents.")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload PDF files first.")

        # Clear chat history button
        if st.button("Clear Chat History"):
            cursor.execute("DELETE FROM chat")
            conn.commit()
            st.session_state["messages"] = []
            st.session_state["doc_content"] = ""
            st.session_state["images"] = []
            st.rerun()

    # Create main chat container
    chat_container = st.container()

    # Display chat history in the chat container
    with chat_container:
        for message in st.session_state["messages"]:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">🧑 {message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', 
                          unsafe_allow_html=True)

    # Create input container at the bottom
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        user_question = st.text_area("Ask about the document content or images:", 
                                   key="user_input", 
                                   height=100)
        
        if st.button("Send", use_container_width=True):
            if user_question:
                # Add user message to conversation
                st.session_state["messages"].append({
                    "role": "user", 
                    "content": user_question
                })

                # Get response using FAISS
                if "faiss_index" in st.session_state and st.session_state["faiss_index"] is not None:
                    relevant_context = retrieve_relevant_chunks(
                        user_question,
                        st.session_state["text_chunks"],
                        st.session_state["faiss_index"]
                    )
                else:
                    relevant_context = st.session_state["doc_content"]

                # Generate response
                with st.spinner("Generating response..."):
                    response = generate_response(relevant_context, user_question)

                # Add bot response to conversation
                st.session_state["messages"].append({
                    "role": "bot", 
                    "content": response
                })

                # Store in database
                cursor.execute(
                    "INSERT INTO chat (user_input, bot_response) VALUES (?, ?)", 
                    (user_question, response)
                )
                conn.commit()

                # Clear the input by updating session state before rerun
                if "user_input" in st.session_state:
                    del st.session_state["user_input"]
                
                # Rerun to update the display
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import streamlit as st
import sqlite3
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API with key from environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
from faster_whisper import WhisperModel
import faiss
from sentence_transformers import SentenceTransformer

# Database setup
conn = sqlite3.connect("video_analysis_chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS chat (user_input TEXT, bot_response TEXT)")
conn.commit()

# Configure Gemini API
model = genai.GenerativeModel("gemini-1.5-pro")

# Load embedding model with explicit CPU device
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Initialize Whisper model
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

def extract_audio_from_video(video_file):
    """Alternative audio extraction using subprocess"""
    try:
        # Create absolute paths for temporary files
        temp_video_path = os.path.abspath("temp_video.mp4")
        
        # Save uploaded file
        with open(temp_video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Use whisper model directly on the video file
        st.info("Processing video...")
        return temp_video_path
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return None

def transcribe_audio(video_path):
    """Transcribes video using faster-whisper."""
    try:
        if video_path is None:
            return ""
        
        st.info("Transcribing content...")
        # Transcribe using faster-whisper
        segments, _ = whisper_model.transcribe(video_path)
        text = " ".join([segment.text for segment in segments])
        
        # Clean up video file
        os.remove(video_path)
        
        return text
    except Exception as e:
        st.error(f"Error transcribing content: {str(e)}")
        if os.path.exists(video_path):
            os.remove(video_path)
        return ""

def store_embeddings_in_faiss(embeddings):
    """Stores embeddings in a FAISS index."""
    try:
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return index
    except Exception as e:
        st.error(f"Error storing embeddings: {str(e)}")
        return None

def retrieve_relevant_chunks(query, embeddings, text_chunks, index):
    """Retrieves relevant transcript chunks based on query."""
    try:
        # Move query embedding to CPU before converting to numpy
        query_embedding = embedder.encode([query], convert_to_tensor=True)
        if hasattr(query_embedding, 'cpu'):
            query_embedding = query_embedding.cpu()
        query_embedding = query_embedding.numpy()
        
        # Ensure query_embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        _, indices = index.search(query_embedding, k=3)
        return " ".join([text_chunks[idx] for idx in indices[0]])
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return ""

def generate_response(context, user_question):
    """Generates response using Gemini Pro."""
    try:
        if not context:
            return "I cannot find relevant information in the video content."
            
        prompt = f"""
        Answer the question based on the extracted video content.
        If the answer cannot be found in the context, say: "I cannot find this information in the video content."

        Context: {context}
        Question: {user_question}

        Answer:
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I couldn't generate a response. Please try again."

def main():
    st.set_page_config("Video Analysis", layout="wide")
    st.title("🎥 Video Analysis Chatbot")

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
            # Get session state values
            index = st.session_state.get("faiss_index")
            text_chunks = st.session_state.get("text_chunks")
            embeddings = st.session_state.get("embeddings")

            # Modified condition check
            if all(x is not None for x in [index, text_chunks, embeddings]):
                try:
                    context = retrieve_relevant_chunks(user_question, embeddings, text_chunks, index)
                    response = generate_response(context, user_question)

                    # Store in database
                    cursor.execute("INSERT INTO chat (user_input, bot_response) VALUES (?, ?)", 
                                 (user_question, response))
                    conn.commit()

                    # Display chat
                    st.markdown(f'<div class="user-message">🧑 {user_question}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="bot-message">🤖 {response}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
            else:
                st.warning("Please process a video first before asking questions.")

    with st.sidebar:
        st.title("Menu:")
        video_file = st.file_uploader("Upload your Video", type=["mp4", "avi", "mov"])
        
        if st.button("Process Video"):
            if video_file:
                with st.spinner("Processing video..."):
                    try:
                        # Process video
                        video_path = extract_audio_from_video(video_file)
                        
                        if video_path:
                            # Transcribe video
                            text = transcribe_audio(video_path)
                            
                            if text:
                                # Split into chunks
                                text_chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
                                
                                # Generate embeddings and move to CPU before numpy conversion
                                embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
                                if hasattr(embeddings, 'cpu'):
                                    embeddings = embeddings.cpu()
                                embeddings = embeddings.numpy()
                                
                                # Store in FAISS
                                index = store_embeddings_in_faiss(embeddings)
                                
                                if index is not None:
                                    # Store in session state
                                    st.session_state["faiss_index"] = index
                                    st.session_state["text_chunks"] = text_chunks
                                    st.session_state["embeddings"] = embeddings
                                    
                                    st.success("Video processed successfully!")
                                else:
                                    st.error("Error creating FAISS index.")
                            else:
                                st.error("No content could be transcribed from the video.")
                        else:
                            st.error("Error processing video.")
                    except Exception as e:
                        st.error(f"Error during video processing: {str(e)}")
            else:
                st.warning("Please upload a video file first.")

        if st.button("Clear Chat History"):
            cursor.execute("DELETE FROM chat")
            conn.commit()
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()

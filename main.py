import streamlit as st
import os
import subprocess

# Define available modules and their descriptions
modules = {
    "csv": {
        "file": "csv_analysis_chatbot.py",
        "keywords": ["csv", "excel", "spreadsheet", "data"],
        "description": "📊 Analyze CSV files and spreadsheets"
    },
    "docx": {
        "file": "docx_analysis_chatbot.py",
        "keywords": ["docx", "doc", "word", "document", "text"],
        "description": "📝 Process Word documents and text files"
    },
    "image": {
        "file": "image_analysis_chatbot.py",
        "keywords": ["image", "picture", "photo", "png", "jpg", "jpeg"],
        "description": "🖼️ Analyze images and photos"
    },
    "pdf": {
        "file": "pdf_analysis_chatbot.py",
        "keywords": ["pdf", "document"],
        "description": "📄 Extract and analyze PDF content"
    },
    "video": {
        "file": "video_analysis_chatbot.py",
        "keywords": ["video", "mp4", "movie", "clip"],
        "description": "🎥 Process and analyze video content"
    }
}

# Custom CSS for a professional UI
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 2rem;
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
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* User message styling */
        .user-message {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        /* Bot message styling */
        .bot-message {
            background-color: #F5F5F5;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

def get_capabilities():
    capabilities = ""
    for module in modules.values():
        capabilities += f"{module['description']}\n"
    return capabilities

import streamlit as st
import os
import subprocess

# Define available modules and their descriptions
modules = {
    "csv": {
        "file": "csv_analysis_chatbot.py",
        "keywords": ["csv", "excel", "spreadsheet", "data", "sheet", "table"],
        "description": "📊 Analyze CSV files and spreadsheets (Excel, data tables)",
    },
    "docx": {
        "file": "docx_analysis_chatbot.py",
        "keywords": ["docx", "doc", "word", "document", "text", "write"],
        "description": "📝 Process Word documents and text files (DOCX, DOC)",
    },
    "image": {
        "file": "image_analysis_chatbot.py",
        "keywords": ["image", "picture", "photo", "png", "jpg", "jpeg", "pic"],
        "description": "🖼️ Analyze images and photos (PNG, JPG, JPEG)",
    },
    "pdf": {
        "file": "pdf_analysis_chatbot.py",
        "keywords": ["pdf", "document", "adobe"],
        "description": "📄 Extract and analyze PDF documents",
    },
    "video": {
        "file": "video_analysis_chatbot.py",
        "keywords": ["video", "mp4", "movie", "clip", "film", "recording"],
        "description": "🎥 Process and analyze video content (MP4, AVI)",
    }
}

# Custom CSS for a professional UI
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 2rem;
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
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* User message styling */
        .user-message {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        /* Bot message styling */
        .bot-message {
            background-color: #F5F5F5;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

def get_module_status(module_name):
    """Check if a module file exists and is accessible"""
    try:
        module_info = modules.get(module_name)
        if module_info and os.path.exists(module_info["file"]):
            return True
        return False
    except:
        return False

def format_capabilities():
    """Format capabilities into a nice string"""
    capabilities = []
    for module_name, info in modules.items():
        status = "✅" if get_module_status(module_name) else "❌"
        capabilities.append(f"{status} {info['description']}")
    return "\n".join(capabilities)

def main():
    st.title("🤖 AI-Powered Analysis Assistant")
    
    # Initialize session state
    if "stage" not in st.session_state:
        st.session_state.stage = "get_name"
    if "name" not in st.session_state:
        st.session_state.name = ""
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Get user's name
    if st.session_state.stage == "get_name":
        st.markdown("### Welcome! 👋")
        name = st.text_input("What's your name?")
        if st.button("Start"):
            if name.strip():
                st.session_state.name = name
                st.session_state.stage = "show_capabilities"
                st.rerun()
            else:
                st.error("Please enter your name to continue.")

    # Show capabilities and main interface
    elif st.session_state.stage == "show_capabilities":
        st.markdown(f"### Hello, {st.session_state.name}! 👋")
        
        # Display capabilities in a nice format
        st.markdown("### I can help you with:")
        for module in modules.values():
            st.markdown(f"- {module['description']}")
        
        st.markdown("### How can I assist you today?")
        
        # Chat interface
        user_input = st.text_input("Type your request here...", key="user_input")
        
        if st.button("Send"):
            if user_input.strip():
                # Add user message to conversation
                st.session_state.conversation.append({"role": "user", "content": user_input})
                
                # Check for keywords and run appropriate module
                found = False
                user_input_lower = user_input.lower()
                
                for module_name, module_info in modules.items():
                    if any(keyword in user_input_lower for keyword in module_info["keywords"]):
                        found = True
                        # Add bot response to conversation
                        st.session_state.conversation.append({
                            "role": "bot", 
                            "content": f"I'll help you with {module_info['description']}..."
                        })
                        
                        # Get the full path of the module file
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        module_path = os.path.join(current_dir, module_info["file"])
                        
                        # Run the appropriate module with error handling
                        try:
                            with st.spinner(f"Running {module_name} analysis..."):
                                # Ensure the file has .py extension
                                if not module_path.endswith('.py'):
                                    module_path += '.py'
                                
                                # Check if file exists
                                if not os.path.exists(module_path):
                                    raise FileNotFoundError(f"Module file {module_path} not found")
                                
                                # Use streamlit run command
                                command = f"streamlit run \"{module_path}\""
                                result = os.system(command)
                                
                                if result == 0:
                                    # Add success message to conversation
                                    st.session_state.conversation.append({
                                        "role": "bot",
                                        "content": f"✅ Launched {module_name.upper()} analysis module. You can return here after you're done."
                                    })
                                else:
                                    raise Exception(f"Failed to run {module_name} module")
                                    
                        except Exception as e:
                            st.error(f"Failed to run {module_name} analysis: {str(e)}")
                            st.session_state.conversation.append({
                                "role": "bot",
                                "content": f"❌ Failed to process your request: {str(e)}"
                            })
                        break
                
                if not found:
                    # Add bot response for unsupported request
                    st.session_state.conversation.append({
                        "role": "bot",
                        "content": """🚫 I apologize, I am an experimental model. Your request is beyond my current capabilities.

                        I can help you with:
                        - 📊 CSV file analysis
                        - 📝 Word document processing
                        - 🖼️ Image analysis
                        - 📄 PDF document analysis
                        - 🎥 Video content analysis

                        How else can I help you?"""
                                            })
                
                st.rerun()

        # Display conversation history
        st.markdown("### Conversation")
        for message in st.session_state.conversation:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="bot-message">
                        <strong>Assistant:</strong> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)

    # Sidebar with additional options
    with st.sidebar:
        st.markdown("### Options")
        if st.button("Start New Conversation"):
            st.session_state.conversation = []
            st.rerun()
        
        if st.button("Change Name"):
            st.session_state.stage = "get_name"
            st.session_state.conversation = []
            st.rerun()

if __name__ == "__main__":
    main()

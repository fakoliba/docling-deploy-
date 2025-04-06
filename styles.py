import streamlit as st

# Define custom avatars
USER_AVATAR = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzYiIGhlaWdodD0iMzYiIHZpZXdCb3g9IjAgMCAzNiAzNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMTgiIGN5PSIxOCIgcj0iMTgiIGZpbGw9IiM0MDQwNDAiLz4KPHBhdGggZD0iTTE4IDIyQzIyLjQxODMgMjIgMjYgMTguNDE4MyAyNiAxNEMyNiA5LjU4MTcyIDIyLjQxODMgNiAxOCA2QzEzLjU4MTcgNiAxMCA5LjU4MTcyIDEwIDE0QzEwIDE4LjQxODMgMTMuNTgxNyAyMiAxOCAyMloiIGZpbGw9IiM1MDUwNTAiLz4KPHBhdGggZD0iTTggMzBDOCAzMCAxMiAyNCAxOCAyNEMyNCAyNCAyOCAzMCAyOCAzMEMyOCAzMCAyNiAzNiAxOCAzNkMxMCAzNiA4IDMwIDggMzBaIiBmaWxsPSIjNTA1MDUwIi8+Cjwvc3ZnPg=="
ASSISTANT_AVATAR = "🐘"

def apply_base_styles():
    """Apply the base styles for the application including Guinea flag theme"""
    st.markdown("""
    <style>
        /* Guinea flag colors - Red, Yellow, Green */
        body {
            background: linear-gradient(to right, #CE1126 33%, #FCD116 33%, #FCD116 66%, #009460 66%);
            margin: 0;
            padding: 0;
        }
        
        /* Style main content area to be readable against colorful background */
        .main .block-container, .stApp .main .block-container {
            background-color: rgba(25, 25, 30, 0.92);
            padding: 1rem;
            border-radius: 10px;
            margin-top: 0;
            padding-top: 0;
            color: white;
        }
        
        /* Remove padding at the top of the page */
        .stApp header {
            background: transparent !important;
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Set header to match the background */
        [data-testid="stHeader"] {
            background: linear-gradient(to right, #CE1126 33%, #FCD116 33%, #FCD116 66%, #009460 66%) !important;
        }
        
        /* Adjust sidebar to be readable */
        .sidebar .sidebar-content {
            background-color: rgba(35, 35, 40, 0.92);
            color: white;
        }
        
        /* Style sidebar elements */
        .sidebar .stButton button {
            background-color: rgba(60, 60, 65, 0.8);
            color: white;
            border: 1px solid rgba(100, 100, 100, 0.5);
        }
        
        .sidebar .stCheckbox label {
            color: white !important;
        }
        
        .sidebar .stSelectbox label {
            color: white !important;
        }
        
        /* Force text color for all elements in the sidebar */
        .sidebar label, .sidebar p, .sidebar span, .sidebar div {
            color: white !important;
        }
        
        /* Style for title */
        .title-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1001;
            text-align: center;
            margin: 0;
            padding: 0.5rem;
            background-color: transparent;
            box-shadow: none;
        }
        
        .title-container h1 {
            color: white;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
            margin: 0;
            padding: 0.5rem;
            display: inline-block;
            border-bottom: 4px solid;
            border-image: linear-gradient(to right, #CE1126 33%, #FCD116 33%, #FCD116 66%, #009460 66%) 1;
        }
        
        /* Adjust chat container */
        .chat-container {
            background-color: transparent;
            height: 90vh; /* Increase height to avoid whitespace */
            padding-top: 4rem;
            display: flex;
            flex-direction: column;
            margin-bottom: 0;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding-bottom: 70px; /* Reduced padding */
            margin-bottom: 0;
        }
        
        .chat-input {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 0.7rem 1rem;
            background-color: rgba(35, 35, 40, 0.95) !important;
            border-top: 4px solid;
            border-image: linear-gradient(to right, #CE1126 33%, #FCD116 33%, #FCD116 66%, #009460 66%) 1;
            z-index: 1000;
            margin-bottom: 0;
        }
        
        /* Style the actual input element */
        .stChatInput {
            background-color: rgba(45, 45, 50, 0.9) !important;
            border: 1px solid rgba(100, 100, 100, 0.5) !important;
            color: white !important;
        }
        
        .stChatInput::placeholder {
            color: rgba(200, 200, 200, 0.7) !important;
        }
        
        /* Style chat messages for better readability */
        .stChatMessage {
            background-color: rgba(50, 50, 50, 0.9) !important;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            margin-bottom: 10px;
            color: white !important;
        }
        
        /* Make sure all text inside chat messages is white for better contrast */
        .stChatMessage p, .stChatMessage span, .stChatMessage div {
            color: white !important;
        }
        
        /* Style code blocks for better visibility */
        .stChatMessage code {
            background-color: rgba(80, 80, 80, 0.9) !important;
            color: #f8f8f8 !important;
            border: 1px solid rgba(100, 100, 100, 0.5);
        }
        
        /* Style user messages to stand out */
        .stChatMessage[data-testid="stChatMessageUser"] {
            border-left: 4px solid #CE1126;
            background-color: rgba(40, 40, 50, 0.9) !important;
        }
        
        /* Style assistant messages */
        .stChatMessage[data-testid="stChatMessageAssistant"] {
            border-left: 4px solid #009460;
            background-color: rgba(35, 45, 40, 0.9) !important;
        }
        
        /* Fix the footer padding */
        footer {
            display: none;
        }
        
        /* Force the background to apply to the entire page */
        .stApp {
            background: linear-gradient(to right, #CE1126 33%, #FCD116 33%, #FCD116 66%, #009460 66%);
        }
        
        /* Make sure the fullscreen wrapper also has the background */
        #root > div:first-child {
            background: linear-gradient(to right, #CE1126 33%, #FCD116 33%, #FCD116 66%, #009460 66%);
        }
    </style>
    """, unsafe_allow_html=True)

def render_app_title():
    """Render the application title with styling"""
    st.markdown('<div class="title-container"><h1>Chat with Sily 🐘</h1></div>', unsafe_allow_html=True)

def initialize_chat_container():
    """Initialize the chat container with proper styling"""
    # Start the flex container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Messages container
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

def close_chat_messages_container():
    """Close the chat messages container"""
    st.markdown('</div>', unsafe_allow_html=True)

def initialize_chat_input():
    """Initialize the chat input container with styling"""
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)

def close_chat_input_container():
    """Close the chat input container"""
    st.markdown('</div>', unsafe_allow_html=True)

def close_chat_container():
    """Close the entire chat container"""
    st.markdown('</div>', unsafe_allow_html=True)

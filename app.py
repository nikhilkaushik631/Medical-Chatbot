import os
import streamlit as st
import google.generativeai as gen_ai
from google.generativeai.types import generation_types
from dotenv import load_dotenv

load_dotenv()
# Configure Streamlit page settings
st.set_page_config(
    page_title="Medical Assistant Chatbot",
    page_icon="🏥",
    layout="centered",
)

# Configure Google Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-1.5-flash')

# Medical assistant instruction prompt
SYSTEM_PROMPT = """You are a professional medical assistant chatbot. Provide original, conversational responses to medical queries while following these guidelines:

1. If the user greets you (e.g., "hello", "good morning"), respond with an appropriate greeting and offer assistance with their medical queries.
2. Give concise, clear answers about medical topics including diseases, medications, symptoms, treatments, procedures, healthcare facilities, hospitals, diet, and health advice.
3. For non-medical questions, respond: "I'm sorry, I can only answer medical-related questions."
4. Structure responses naturally with:
   - Brief overview of the topic
   - Key relevant details if asked
   - Practical advice when appropriate
5. Use your own words and avoid quoting or closely paraphrasing other sources.
"""

# Function to handle message sending with error handling
def send_message_safely(chat_session, message):
    try:
        response = chat_session.send_message(message)
        return response.text
    except generation_types.StopCandidateException:
        return ("I apologize, but I need to rephrase my response to ensure originality. "
                "Please ask your question again, and I'll provide a fresh, unique answer.")
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try asking your question again."

# Initialize chat session with system prompt
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
    # Send the system prompt once at the start
    send_message_safely(st.session_state.chat_session, SYSTEM_PROMPT)

# Display the chatbot's title on the page
st.title("Medical Assistant Chatbot")

# Function to translate roles between Gemini and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# Display the chat history (excluding the system prompt)
for message in st.session_state.chat_session.history[1:]:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# Input field for user's message
user_prompt = st.chat_input("Ask a medical question...")

if user_prompt:
    # Display user's message
    st.chat_message("user").markdown(user_prompt)
    
    # Send user's message with error handling
    response_text = send_message_safely(st.session_state.chat_session, user_prompt)
    
    # Display Gemini's response
    with st.chat_message("assistant"):
        st.markdown(response_text)

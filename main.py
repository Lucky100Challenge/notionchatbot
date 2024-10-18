import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up the model
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Set up the conversation memory
memory = ConversationBufferMemory(return_messages=True)

# Create a custom prompt template
template = """You are a helpful AI assistant embedded in a Notion page. Be concise, friendly, and helpful.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# Set up the conversation chain
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Streamlit UI
st.title("Notion AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI response
    response = conversation.predict(input=user_input)

    # Display AI response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a footer with information about the chatbot
st.markdown("---")
st.markdown("This AI chatbot is powered by Google's Generative AI and LangChain.")

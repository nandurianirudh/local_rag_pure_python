import streamlit as st
from py_rag import llm_chatbot_constitution
from azure.ai.inference.models import UserMessage, AssistantMessage

st.set_page_config(page_title="BITSy RAG Chatbot", page_icon="🤖", layout="centered")

# --- Initialize chatbot in session ---
if "chatbot" not in st.session_state:
    st.session_state.chatbot = llm_chatbot_constitution()
    st.session_state.chatbot.history = []

st.title("📜 Student Constitution Chatbot (BITSy)")
st.caption("Ask questions about the student constitution, committees, or election rules.")

# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User input ---
if prompt := st.chat_input("Ask your question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    bot = st.session_state.chatbot

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            funky_op = bot.funky_answer_generator(prompt)

            if funky_op["can_answer_from_system_info"]:
                response = funky_op["answer"]
            else:
                cleaned_info = bot.clean_user_message(prompt)
                cleaned_query = cleaned_info["cleaned_question"]

                if cleaned_query.startswith("Just to clarify"):
                    response = cleaned_query
                elif "I dont have enough information" in cleaned_query:
                    response = cleaned_query
                else:
                    bot.section = cleaned_info["section_name"]
                    context = bot.get_context(cleaned_query)
                    response = bot.rag_answer(cleaned_query, context)

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
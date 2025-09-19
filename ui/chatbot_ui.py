# ui/chatbot_ui.py
import streamlit as st

class ChatbotInterface:
    def __init__(self):
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

    def render(self):
        st.subheader("🤖 AI Trading Assistant Chatbot")

        for msg in st.session_state['chat_history']:
            if msg['sender'] == 'user':
                st.markdown(f"**You:** {msg['message']}")
            else:
                st.markdown(f"**Assistant:** {msg['message']}")

        user_input = st.text_input("Ask a trading question:")

        if user_input:
            st.session_state['chat_history'].append({'sender': 'user', 'message': user_input})
            # Here you would call your chatbot to generate a response
            bot_response = "This is a placeholder response."
            st.session_state['chat_history'].append({'sender': 'bot', 'message': bot_response})

            st.experimental_rerun()

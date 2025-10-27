import streamlit as st
import requests
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="GenAI Chat", page_icon="🤖", layout="centered")

st.title("💬 GenAI Chat com Memória (LangChain)")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Caixa de input para o utilizador
user_input = st.text_input("Escreve a tua mensagem:", "")

# Botão para enviar mensagem
if st.button("Enviar"):
    if user_input.strip() != "":
        payload = {"session_id": st.session_state.session_id, "message": user_input}
        try:
            response = requests.post(f"{BACKEND_URL}/chat", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.session_state.last_response = data["answer"]
            else:
                st.error(f"Erro {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Erro ao ligar ao backend: {e}")

# Mostrar resposta do modelo
if "last_response" in st.session_state:
    st.markdown("**Resposta:**")
    st.write(st.session_state.last_response)

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

load_dotenv()

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    user_agent="lekta-rag/0.1"
)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)
llm = init_chat_model("meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq")

SYSTEM_PROMPT = (
    "Eres un asistente experto en responder preguntas usando solo la información proporcionada.\n"
    "Tu misión es maximizar la utilidad al usuario, sin desviarte jamás del contexto.\n\n"

    "OBJETIVOS:\n"
    "1. Responder con precisión y brevedad (≤ 40 palabras).\n"
    "2. Ayudar al usuario al máximo con la información disponible.\n\n"

    "RESTRICCIONES:\n"
    "- No menciones la fuente, el contexto ni uses expresiones tipo “según…”, “documentación”.\n"
    "- No especules, conjetures ni inventes datos.\n"
    "- No uses saludos, despedidas ni frases de cortesía.\n"
    "- Si no hay información suficiente, responde EXACTAMENTE:\n"
    "  “No hay información disponible para responder a esta pregunta.”\n\n"

    "FORMATO:\n"
    "- Texto plano, máximo 40 palabras.\n"
    "- Si aportas listas o viñetas, que sean muy breves (≤ 3 ítems).\n\n"

    "Pregunta: {question}\n"
    "Información: {context}\n"
    "Respuesta:"
)

question = st.text_input("Introduce tu pregunta")
k = st.slider("Fragmentos a recuperar (k)", 1, 10, 4)

if st.button("Consultar") and question.strip():
    with st.spinner("Buscando respuesta..."):
        start = time.time()
        docs = vector_store.similarity_search(question, k=k)
        if not docs:
            st.warning("No se recuperaron fragmentos.")
        else:
            context = "\n\n".join(d.page_content[:1000] for d in docs)
            prompt = SYSTEM_PROMPT.format(question=question, context=context)
            answer = llm.invoke(prompt).content
            st.success(answer)
            st.caption(f"⏱️ Tiempo: {round(time.time() - start, 3)}s")
            with st.expander("Fragmentos usados"):
                for d in docs:
                    st.markdown(d.page_content)

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

load_dotenv()

if not os.getenv("COHERE_API_KEY"):
    st.error("⚠️ Falta la variable de entorno COHERE_API_KEY.")
    st.stop()

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

with st.form("rag_form"):
    question = st.text_input("Pregunta")
    k = st.slider("Fragmentos a recuperar", 1, 20, 4)
    submitted = st.form_submit_button("Consultar")

if submitted and question.strip():
    with st.spinner("Buscando respuesta..."): 
        start = time.time()
        docs = vector_store.max_marginal_relevance_search(query=question, k=k, fetch_k=20)

        if not docs:
            st.warning("No se recuperaron fragmentos.") 
        else:
            max_chars = 3000
            context = ""
            for d in docs:
                if len(context) + len(d.page_content) <= max_chars:
                    context += d.page_content + "\n\n"
                else:
                    break

            prompt = SYSTEM_PROMPT.format(question=question, context=context)

            try:
                answer = llm.invoke(prompt).content
                st.success(answer)
            except Exception as e:
                st.error(f"Error al invocar el modelo: {str(e)}")
                st.stop()

            st.caption(f"⏱️ Tiempo: {round(time.time() - start, 3)}s")

            st.markdown("### Fragmentos usados")
            for i, d in enumerate(docs, 1):
                with st.expander(f"Fragmento {i}", expanded=False):
                    st.markdown(d.page_content)

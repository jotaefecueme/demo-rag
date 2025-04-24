import os
import time
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    user_agent="lekta-rag/0.1"
)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

llm = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct",
    model_provider="groq"
)

SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas exclusivamente con la información proporcionada.\n"
    "Tu respuesta debe ser clara, concisa (máx. 30 palabras) y ceñida estrictamente a ese contexto.\n\n"
    "INSTRUCCIONES OBLIGATORIAS:\n"
    "- Bajo ningún concepto menciones la fuente, el contexto ni palabras como “documentación”, “texto”, “según…”.\n"
    "- No especules ni rellenes lagunas.\n"
    "- No uses frases genéricas ni fórmulas de cortesía.\n"
    "- Si no hay datos para responder, responde EXACTAMENTE:\n"
    "  “No hay información disponible para responder a esta pregunta.”\n\n"
    "Pregunta: {question}\n"
    "Contexto: {context}\n"
    "Respuesta:"
)

def retrieve_context(question: str, k: int = 4):
    return vector_store.similarity_search(question, k=k)

def build_prompt(question: str, documents: list[Document], max_length=1000):
    context = "\n\n".join(doc.page_content[:max_length] for doc in documents)  
    return SYSTEM_PROMPT.format(question=question, context=context)

def get_response(prompt: str):
    return llm.invoke(prompt).content

st.sidebar.markdown("### Configuración")
k = st.sidebar.slider("Número de fragmentos recuperación (k)", min_value=1, max_value=20, value=4)

question = st.text_input("Pregunta:", "")
docs = []
response = ""
elapsed = None

if st.button("Consultar") and question.strip():
    try:
        start_time = time.time()

        docs = retrieve_context(question, k)
        if not docs:
            st.warning("No se recuperaron fragmentos. ¿Seguro que has indexado documentos?")
        else:
            prompt = build_prompt(question, docs)
            with st.spinner("Generando respuesta..."):
                response = get_response(prompt)
                elapsed = round(time.time() - start_time, 4)

            st.write(response if response else "No se obtuvo respuesta del modelo.")
            st.markdown(f"**Tiempo de respuesta:** `{elapsed} segundos`")

            st.markdown("### Fragmentos recuperados")
            for i, doc in enumerate(docs, start=1):
                with st.expander(f"Fragmento {i}"):
                    st.write(doc.page_content)

    except Exception as e:
        logging.error(f"Error al procesar la consulta: {e}")
        st.error("Se ha producido un error al procesar la consulta.")

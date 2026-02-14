import streamlit as st

from src.config import load_settings


def main() -> None:
    load_settings()

    st.set_page_config(page_title="RAG Normativa Laboral", layout="wide")
    st.title("RAG Normativa Laboral Colombiana")

    user_query = st.text_input("Consulta", placeholder="Escribe tu pregunta...")
    submit = st.button("Consultar")

    if submit:
        # Placeholder outputs until RAG logic is implemented.
        st.subheader("Respuesta")
        st.write("(pendiente de implementacion)")

        st.subheader("Documentos recuperados")
        st.write("(pendiente de implementacion)")

        st.subheader("Metadata")
        st.write("(pendiente de implementacion)")

        st.subheader("Logs de verificacion")
        st.write("(pendiente de implementacion)")


if __name__ == "__main__":
    main()

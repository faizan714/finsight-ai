
import streamlit as st
from main import (
    load_pdf,
    split_documents,
    create_vector_store,
    ask_financial_question
)

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 FinSight AI")
st.subheader("AI-Powered Financial Report Analyst")

st.write(
    "Upload a financial report and ask questions "
    "about its financial performance."
)

uploaded_file = st.file_uploader(
    "Upload Financial PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Process Financial Report"):

        with st.spinner("Processing your financial report..."):

            import fitz

            pdf_document = fitz.open(
                stream=uploaded_file.getvalue(),
                filetype="pdf"
            )

            from langchain_core.documents import Document

            pages = [
                Document(
                    page_content=page.get_text(),
                    metadata={
                        "page_number": page.number + 1,
                        "source": uploaded_file.name
                    }
                )
                for page in pdf_document
            ]

            chunks = split_documents(pages)

            vector_store = create_vector_store(chunks)

            st.session_state.vector_store = vector_store

            st.success("Financial report processed successfully!")

if "vector_store" in st.session_state:

    st.divider()

    st.subheader("Ask Your Financial Questions")

    question = st.text_input(
        "Enter your question:",
        placeholder="How much revenue did the company earn in 2025?"
    )

    if st.button("Analyze Report"):

        if question.strip():

            with st.spinner("Analyzing your question..."):

                answer, source_docs = ask_financial_question(
                    st.session_state.vector_store,
                    question
                )

            st.markdown("### AI Answer")

            if isinstance(answer, list):
                for item in answer:
                    if item.get("type") == "text":
                        st.markdown(item["text"])
            else:
                st.markdown(answer)

            st.markdown("### Source Pages")

            source_pages = sorted({
                doc.metadata["page_number"]
                for doc in source_docs
            })

            st.write(", ".join(map(str, source_pages)))

        else:
            st.warning("Please enter a question.")

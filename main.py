
import os
import fitz
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

PDF_PATH = Path("documents/financial_report.pdf")


def load_pdf(pdf_path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path.resolve()}")

    documents = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page_number": page_number,
                            "source": pdf_path.name,
                        },
                    )
                )

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True,
    )
    return splitter.split_documents(documents)




def create_vector_store(chunks):
    from pathlib import Path
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="SEMANTIC_SIMILARITY",
        output_dimensionality=256,
    )

    index_path = "faiss_index"

    # Load existing vector database
    if Path(index_path, "index.faiss").exists():
        print("Loading existing FAISS database...")
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Create and save a new vector database
    else:
        print("Creating a new FAISS database...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)
        print("FAISS database saved successfully!")

    return vector_store



def ask_financial_question(vector_store, question):

    # Step 1: Retrieve relevant chunks
    relevant_docs = vector_store.similarity_search(
        question,
        k=3
    )

    # Step 2: Prepare context
    context = "\n\n".join(
        [
            f"Page {doc.metadata['page_number']}:\n{doc.page_content}"
            for doc in relevant_docs
        ]
    )

    # Step 3: Create prompt
    prompt = ChatPromptTemplate.from_template("""
    You are FinSight AI, a financial report analyst.

    Answer the user's question using ONLY the
    financial information provided below.

    If the answer is not available in the context,
    clearly say that the report does not contain
    enough information.

    Always mention the source page number.

    Financial Report Context:
    {context}

    User Question:
    {question}

    Answer:
    """)

    # Step 4: Connect Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2
    )

    # Step 5: Generate answer
    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content, relevant_docs


if __name__ == "__main__":

    pages = load_pdf(PDF_PATH)
    chunks = split_documents(pages)

    print(f"Pages loaded: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    vector_store = create_vector_store(chunks)

    print("\nFinSight AI is ready!")

    question = "How much revenue did the company earn in 2025?"

    answer, sources = ask_financial_question(
        vector_store,
        question
    )

    print("\nQuestion:", question)
    print("\nFinSight AI Answer:")
    if isinstance(answer, list):
        print("\n".join(item["text"] for item in answer if item.get("type") == "text"))
    else:
        print(answer)

    print("\nSource Pages:")

    for doc in sources:
        print(f"Page {doc.metadata['page_number']}")




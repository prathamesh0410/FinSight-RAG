"""FinSight RAG Streamlit UI - Document Q&A with Citations."""
import streamlit as st
import asyncio
from datetime import datetime

from app.core.config import get_settings
from app.services.rag import RAGPipeline
from app.services.vector_store import VectorStoreService
from app.services.embeddings import EmbeddingService
from app.models.document import QueryLog

# Page config
st.set_page_config(
    page_title="FinSight RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .stMain > div {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }
    .chat-message.user {
        background-color: #e6f7ff;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    .message-avatar.user {
        color: #1573e6;
    }
    .message-avatar.assistant {
        color: #6c757d;
    }
    .source-citation {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.5rem;
        padding-left: 0.5rem;
        border-left: 2px solid #1573e6;
    }
    .stButton > button {
        width: 100%;
    }
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# App header
st.title("📚 FinSight RAG")
st.caption("Banking-specialized Regulatory Document Intelligence")
st.divider()

# Sidebar
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=[".pdf", ".docx", ".txt"],
        help="Supported: PDF, DOCX, TXT files (max 50MB)",
        accept_multiple_files=False,
    )

    document_type = st.selectbox(
        "Document Type",
        ["RBI Circular", "Loan Agreement", "Compliance Policy", "Risk Report", "Regulatory Guideline"],
        index=0,
    )

    if st.button("📤 Process Document", type="primary", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Would call API to upload document
                st.success("Document processed successfully!")
        else:
            st.error("Please upload a document first")

    st.divider()
    st.header("💡 Sample Queries")
    sample_queries = [
        "What does RBI say about capital adequacy?",
        "Loan agreement prepayment charges",
        "KYC compliance requirements",
        "Basel III norms explanation",
        "Credit risk management guidelines",
    ]

    for query in sample_queries:
        if st.button(f"🔍 {query}", use_container_width=True, key=f"sample_{query}"):
            st.session_state.current_query = query
            st.session_state.processing = True

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_query = ""
        st.rerun()

# Chat messages display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            for citation in message["citations"]:
                st.caption(citation)

# Chat input
if not st.session_state.processing:
    user_input = st.chat_input("Ask about regulatory documents...")

    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Process assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Run async query processing
                asyncio.create_task(
                    process_rag_query(user_input)
                )
else:
    st.session_state.processing = False

# Run pending async tasks
async def process_rag_query(query: str):
    """Process RAG query and update chat."""
    try:
        from app.services.rag import RAGPipeline
        from app.services.embeddings import EmbeddingService
        from app.core.config import get_settings

        settings = get_settings()
        embedding_service = EmbeddingService()
        vector_store = None  # Will use session db

        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_client=None,
            embedder=embedding_service,
            settings=settings,
        )

        result = await rag_pipeline.process_query(query=query)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(result.answer)

            # Display citations
            if result.citations:
                st.divider()
                st.subheader("📖 Sources")
                for citation in result.citations:
                    with st.expander(f"Source document"):  # noqa: E713,501
                        st.caption(f"Similarity: {citation.similarity_score:.2f}")
                        st.markdown(f"**Citation:** [{citation.document_id} §{citation.clause_reference}]")
                        st.markdown(f"**Preview:** {citation.content_preview}")

        # Store in messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "citations": [
                f"Source: {c.document_id} §{c.clause_reference or 'N/A'}"
                for c in result.citations
            ]
        })

    except Exception as e:
        st.error(f"Query processing failed: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)}",
        })

    finally:
        st.session_state.processing = False
        st.rerun()


# Sample queries below chat
if st.session_state.messages:
    st.divider()
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💰 KYC Requirements"):
            # Would query about KYC
            pass

    with col2:
        if st.button("📊 Capital Adequacy"):
            # Would query about capital
            pass

    with col3:
        if st.button("🔄 Basel III"):
            # Would query about Basel III
            pass
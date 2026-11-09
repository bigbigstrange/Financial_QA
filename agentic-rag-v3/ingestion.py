import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Keep HuggingFace cache on the project drive (avoid full C: disk) and use China mirror by default
_hf_home = Path(__file__).resolve().parent / ".cache" / "huggingface"
_hf_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf_home))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embedding_function,
).as_retriever()


if __name__ == "__main__":
    sample_docs = [
        Document(
            page_content="Agent memory refers to the mechanisms that allow AI agents to store, retrieve, and use past information during interactions. There are several types of agent memory: short-term memory (within a single conversation), long-term memory (across sessions), and working memory (active reasoning state). Techniques like retrieval-augmented generation (RAG), vector databases, and episodic memory buffers are commonly used to implement agent memory systems.",
            metadata={"source": "agent-memory-intro"}
        ),
        Document(
            page_content="Prompt engineering is the practice of designing and optimizing prompts to guide large language models (LLMs) toward desired outputs. Key techniques include: few-shot prompting (providing examples), chain-of-thought prompting (encouraging step-by-step reasoning), instruction tuning, and prompt templates. Advanced methods like automatic prompt optimization and tree-of-thoughts prompting have emerged to improve LLM performance on complex reasoning tasks.",
            metadata={"source": "prompt-engineering-guide"}
        ),
        Document(
            page_content="Adversarial attacks on language models involve crafting inputs that cause the model to produce harmful, biased, or unintended outputs. Common attack types include: prompt injection (overriding system instructions), jailbreaking (bypassing safety guardrails), data poisoning (contaminating training data), and membership inference attacks. Defenses include input sanitization, constitutional AI, RLHF (reinforcement learning from human feedback), and red-teaming to identify vulnerabilities before deployment.",
            metadata={"source": "adversarial-attacks-overview"}
        ),
        Document(
            page_content="Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. The process involves: 1) receiving a user query, 2) retrieving relevant documents from a knowledge base using embeddings and vector search, 3) augmenting the LLM prompt with retrieved context, and 4) generating a response grounded in the retrieved documents. RAG helps reduce hallucinations and keeps responses factually accurate by providing up-to-date external knowledge to the model.",
            metadata={"source": "rag-explained"}
        ),
        Document(
            page_content="LangGraph is a framework for building stateful, multi-actor applications with LLMs using graph-based workflows. It allows developers to define nodes (processing steps) and edges (transitions) to create complex agent workflows. Key features include: conditional routing, state management across nodes, cyclical graphs for iterative reasoning loops, and integration with LangChain tools and models. LangGraph is particularly useful for building autonomous agents with reflection, tool use, and self-correction capabilities.",
            metadata={"source": "langgraph-overview"}
        ),
    ]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(sample_docs)

    Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding_function,
        persist_directory="./.chroma",
    )

    print(f"Ingestion complete: {len(doc_splits)} chunks indexed into Chroma vectorstore.")

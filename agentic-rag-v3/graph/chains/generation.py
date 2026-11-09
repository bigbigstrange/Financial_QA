from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1", temperature=0)

# Equivalent to LangChain Hub prompt "rlm/rag-prompt" (inlined to avoid hub import breakage)
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:"""
)

generation_chain = prompt | llm | StrOutputParser()

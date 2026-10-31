# import os

# from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import (
#     RunnablePassthrough,
#     RunnableLambda
# )

# from core.vector_store import (
#     build_vector_store,
#     load_vector_store,
#     get_retriever
# )


# # Load environment variables
# load_dotenv()


# # 1. Get OpenRouter LLM
# def get_llm():

#     api_key = os.getenv("OPENROUTER_API_KEY")
#     model = os.getenv("OPENROUTER_MODEL")

#     if not api_key:
#         raise ValueError("OPENROUTER_API_KEY is missing in .env")

#     if not model:
#         raise ValueError("OPENROUTER_MODEL is missing in .env")

#     return ChatOpenAI(
#         model=model,
#         api_key=api_key,
#         base_url="https://openrouter.ai/api/v1",
#         temperature=0.3
#     )


# # 2. Format Retrieved Documents
# def format_docs(docs):

#     return "\n\n".join(
#         [doc.page_content for doc in docs]
#     )


# # 3. Build RAG Chain (New Transcript)
# def build_rag_chain(transcript: str):

#     # Build vector database
#     vector_store = build_vector_store(transcript)

#     # Create retriever
#     retriever = get_retriever(vector_store, k=4)

#     # Get LLM
#     llm = get_llm()

#     # Prompt Template
#     prompt = ChatPromptTemplate.from_messages([

#         (
#             "system",
#             """You are an expert meeting assistant.

# Answer the user's question based ONLY on the meeting
# transcript context provided below.

# If the answer is not found in the context, say:
# "I could not find this information in the meeting transcript."

# Rules:
# - Be concise and precise.
# - Do not invent information.
# - If quoting someone, mention it clearly.
# - Answer according to the transcript.

# Context from meeting transcript:
# {context}"""
#         ),

#         ("human", "{question}")

#     ])

#     # LCEL RAG Pipeline
#     rag_chain = (

#         {
#             "context": retriever | RunnableLambda(format_docs),

#             "question": RunnablePassthrough()
#         }

#         | prompt
#         | llm
#         | StrOutputParser()

#     )

#     return rag_chain


# # 4. Load Existing RAG Chain
# def load_rag_chain():

#     # Load existing vector database
#     vector_store = load_vector_store()

#     # Create retriever
#     retriever = get_retriever(vector_store, k=4)

#     # Get LLM
#     llm = get_llm()

#     # Prompt Template
#     prompt = ChatPromptTemplate.from_messages([

#         (
#             "system",
#             """You are an expert meeting assistant.

# Answer the user's question based ONLY on the meeting
# transcript context provided below.

# If the answer is not found in the context, say:
# "I could not find this information in the meeting transcript."

# Rules:
# - Be concise and precise.
# - Do not invent information.
# - If quoting someone, mention it clearly.
# - Answer according to the transcript.

# Context from meeting transcript:
# {context}"""
#         ),

#         ("human", "{question}")

#     ])

#     # LCEL RAG Pipeline
#     rag_chain = (

#         {
#             "context": retriever | RunnableLambda(format_docs),

#             "question": RunnablePassthrough()
#         }

#         | prompt
#         | llm
#         | StrOutputParser()

#     )

#     return rag_chain


# # 5. Ask Question
# def ask_question(rag_chain, question: str) -> str:

#     print(f"\nQuestion: {question}")

#     answer = rag_chain.invoke(question)

#     print(f"\nAnswer: {answer}")

#     return answer



import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import build_vector_store, get_retriever

load_dotenv()


def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing in .env")

    if not model:
        raise ValueError("OPENROUTER_MODEL missing in .env")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert video assistant. Answer the user's question
based ONLY on the video transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the video."

Always be concise and precise.

Context from video transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    # Full LCEL RAG pipeline
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    answer = rag_chain.invoke(question)
    return answer

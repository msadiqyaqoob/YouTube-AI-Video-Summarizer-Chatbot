
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()


# Same LLM configuration as summarizer.py
def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing in .env")

    if not model:
        raise ValueError("OPENROUTER_MODEL is missing in .env")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2
    )


# Reusable chain builder
def build_chain(system_prompt: str):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    return prompt | llm | StrOutputParser()


# 1. Extract Action Items
def extract_action_items(transcript: str) -> str:

    chain = build_chain(
        "You are an expert meeting analyst. "
        "Analyze the meeting transcript and extract all actionable items.\n\n"

        "For each action item provide:\n"
        "1. Task description\n"
        "2. Owner (person responsible)\n"
        "3. Deadline (if mentioned, otherwise write 'Not specified')\n\n"

        "Rules:\n"
        "- Only extract explicitly discussed action items.\n"
        "- Do not invent owners or deadlines.\n"
        "- Format as a numbered list.\n"
        "- If none found, say 'No action items found.'"
    )

    return chain.invoke({"text": transcript})


# 2. Extract Key Decisions
def extract_key_decisions(transcript: str) -> str:

    chain = build_chain(
        "You are an expert meeting analyst. "
        "Analyze the meeting transcript and extract all important decisions.\n\n"

        "Rules:\n"
        "- Extract decisions that were actually agreed upon.\n"
        "- Include relevant context where necessary.\n"
        "- Do not confuse suggestions with final decisions.\n"
        "- Do not invent information.\n"
        "- Format as a numbered list.\n"
        "- If none found, say 'No key decisions found.'"
    )

    return chain.invoke({"text": transcript})


# 3. Extract Open Questions
def extract_questions(transcript: str) -> str:

    chain = build_chain(
        "You are an expert meeting analyst. "
        "Analyze the meeting transcript and extract unresolved questions "
        "or topics that require further discussion.\n\n"

        "Rules:\n"
        "- Include unanswered questions.\n"
        "- Include topics requiring clarification or follow-up.\n"
        "- Do not include questions that were already answered.\n"
        "- Do not invent information.\n"
        "- Format as a numbered list.\n"
        "- If none found, say 'No open questions found.'"
    )

    return chain.invoke({"text": transcript})
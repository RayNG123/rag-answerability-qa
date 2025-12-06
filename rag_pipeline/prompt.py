"""
Prompt templates and helper to build prompts for different RAG modes.
"""

from __future__ import annotations

from typing import Iterable

PROMPT_WITH_CITATIONS = """
You are a question answering assistant. You will be given a question and several paragraphs of context.

Each paragraph is labeled with an index like [1], [2], [3], etc.

Respond with **JSON only** matching:
{{
  "citations": [string]    // paragraph indices you used (e.g., ["1","3"]); empty list if not answerable
  "answer": string    // concise answer; empty string if not answerable
}}

Rules:
- Use ONLY the provided paragraphs to answer the given questions, and some of the questions might be unanswerable.
- Answer and cite only if you are certain based on the provided paragraphs.
- Keep the answer concise (1–3 sentences) when answerable.
- When you cite, list paragraph indices as list of strings (e.g., ["1","3"]).
- If not enough info and unanswerable, set answer="", citations=[].

Context:
{context_block}

Question:
{question}

JSON Response:
"""

PROMPT_NO_CITATIONS = """
You are a question answering assistant. You will be given a question and several paragraphs of context.

Each paragraph is labeled with an index like [1], [2], [3], etc.

Respond with **JSON only** matching:
{{
  "answerable": boolean,   // true if the provided paragraphs contain enough information to answer; false otherwise
  "answer": string         // concise answer; empty string if not answerable
}}

Rules:
- Use ONLY the provided paragraphs.
- Some questions may be unanswerable; mark them as such if the context does not provide sufficient information.
- Provide an answer only when you are confident it is fully supported by the paragraphs.
- Keep the answer concise (1–3 sentences) when answerable.
- If unanswerable, set "answerable": false and "answer": "".

Context:
{context_block}

Question:
{question}

JSON Response:
"""


PROMPT_CITATION_ONLY = """
You are a question answering assistant. You will be given a question and several paragraphs of context.

Each paragraph is labeled with an index like [1], [2], [3], etc.

Your task is to identify which paragraphs could be used to support a correct answer to the question.

Respond with **JSON only** matching:
{{
  "citations": [string]    // paragraph indices you would rely on (e.g., ["1","3"]); empty if the context cannot support an answer
}}

Only include indices for paragraphs that provide direct or strong supporting evidence for the answer.
If none of given context is sufficient to support any answer, return "citations": [].

Context:
{context_block}

Question:
{question}

JSON Response:
"""


PROMPT_ANSWER_ONLY = """
You are a question answering assistant. You will be given a question and several paragraphs of context.

Each paragraph is labeled with an index like [1], [2], [3], etc.

Your task is to provide a concise answer to the question based solely on the provided context.

Respond with **JSON only** matching:
{{
  "answer": string    // concise answer; empty string if not answerable
}}

Rules:
- Use ONLY the provided paragraphs to answer the given questions

Context:
{context_block}

Question:
{question}

JSON Response:
"""


PROMPT_ANSWERABLE_ONLY = """
You are a question answering assistant. You will be given a question and several paragraphs of context.

Each paragraph is labeled with an index like [1], [2], [3], etc.

Your task is to determine if the provided paragraphs contain enough information to answer the question.

Respond with **JSON only** matching:
{{
  "answerable": boolean,   // true if the provided paragraphs contain enough information to answer; false otherwise
}}

Rules:
- Use ONLY the provided paragraphs.
- If none of given context is sufficient to support any answer, return "answerable": false.

Context:
{context_block}

Question:
{question}

JSON Response:
"""


def build_rag_prompt(
    question: str,
    context_paragraphs: Iterable[str],
    mode: str = "with_citations",
) -> str:
    """
    Build a RAG prompt based on the specified mode.

    Args:
        question (str): The question to be answered.
        context_paragraphs (Iterable[str]): The context paragraphs to use.
        mode (str): The RAG mode. One of "with_citations", "no_citations", "citation_only", "answer_only".

    Returns:
        str: The constructed prompt.
    """
    if mode == "with_citations":
        prompt_template = PROMPT_WITH_CITATIONS
        context_block = "\n\n".join(
        f"[{i+1}] {para}" for i, para in enumerate(context_paragraphs)
        )
        return prompt_template.format(
          context_block=context_block,
          question=question
        )
    elif mode == "no_citations":
        prompt_template = PROMPT_NO_CITATIONS
        context_block = "\n\n".join(
        f"[{i+1}] {para}" for i, para in enumerate(context_paragraphs)
        )
        return prompt_template.format(
          context_block=context_block,
          question=question
        )
    elif mode == "citation_only":
        prompt_template = PROMPT_CITATION_ONLY
        context_block = "\n\n".join(
        f"[{i+1}] {para}" for i, para in enumerate(context_paragraphs)
        )
        return prompt_template.format(
          context_block=context_block,
          question=question
        )
    elif mode == "answer_only":
        prompt_template = PROMPT_ANSWER_ONLY
        context_block = "\n\n".join(
        f"[{i+1}] {para}" for i, para in enumerate(context_paragraphs)
        )
        return prompt_template.format(
          context_block=context_block,
          question=question
        )
    elif mode == "answerable_only":
        prompt_template = PROMPT_ANSWERABLE_ONLY
        context_block = "\n\n".join(
        f"[{i+1}] {para}" for i, para in enumerate(context_paragraphs)
        )
        return prompt_template.format(
          context_block=context_block,
          question=question
        )
    else:
        raise ValueError(f"Unsupported RAG mode: {mode}")

    
"""
Evaluation helpers: parsing model answers, normalization, metrics, citation mapping, and judge prompt.
"""

from __future__ import annotations

import ast
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
import re


def map_citation_indices_to_doc_ids(
    indices: Iterable[int],
    docs: List[Dict[str, Any]],
) -> List[str]:
    """
    Map 1-based paragraph indices to doc_ids in a retrieved_docs list.

    docs[i] is assumed to have key 'doc_id'; index i+1 corresponds to [i+1] in text.
    """
    mapped: List[str] = []
    for idx in indices or []:
        try:
            i = int(idx)
        except Exception:
            continue
        if 1 <= i <= len(docs):
            doc_id = docs[i - 1].get("doc_id")
            if doc_id is not None:
                mapped.append(str(doc_id))
    return mapped


def _extract_json_substring(raw: Any) -> Optional[str]:
    """
    Try to extract the first valid-looking JSON object substring from raw text
    by tracking balanced braces.

    Useful when the model wraps the JSON in extra commentary.
    """
    if raw is None:
        return None

    try:
        text = str(raw)
    except Exception:
        return None

    if not text:
        return None

    start_idx: Optional[int] = None
    depth = 0

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    # First complete JSON object
                    return text[start_idx : i + 1]

    return None


import json
import ast
import re
from typing import Any, Dict, Optional


def _extract_json_substring(raw: Any) -> Optional[str]:
    """
    Try to extract the first valid-looking JSON object substring from raw text
    by tracking balanced braces. If no balanced object is found (e.g., truncated),
    fall back to everything from the first '{' to the end.
    """
    if raw is None:
        return None

    try:
        text = str(raw)
    except Exception:
        return None

    if not text:
        return None

    start_idx: Optional[int] = None
    depth = 0

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    # First complete JSON object
                    return text[start_idx : i + 1]

    # No balanced object found; fall back to "from first { to end"
    first = text.find("{")
    if first == -1:
        return None
    return text[first:]


def _load_json_dict(raw_answer: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort: convert raw_answer to string, extract JSON substring, and parse into a dict.
    Returns None on any failure.
    """
    json_sub = _extract_json_substring(raw_answer)
    if json_sub is None:
        return None

    def _try_json(s: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(s)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    # 1) Straight JSON parse
    data = _try_json(json_sub)
    if data is not None:
        return data

    # 2) Lenient cleanup: strip, remove trailing commas before } or ]
    s_clean = json_sub.strip()
    s_clean = re.sub(r",\s*([}\]])", r"\1", s_clean)

    data = _try_json(s_clean)
    if data is not None:
        return data

    # 3) Try Python literal eval as a fallback for slightly-off JSON
    try:
        obj = ast.literal_eval(s_clean)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 4) Last resort: manually extract known fields (e.g., "answer" and "citations")
    out: Dict[str, Any] = {}

    # citations: "citations": ["1","2"] or 'citations': ['1','2']
    m_cite = re.search(r'["\']citations["\']\s*:\s*\[([^\]]*)\]', s_clean)
    if m_cite:
        raw_list = m_cite.group(1)
        citations = [
            x.strip().strip('"').strip("'")
            for x in raw_list.split(",")
            if x.strip().strip('"').strip("'")
        ]
        if citations:
            out["citations"] = citations

    # Try to extract "answer" robustly, even if closing quote/brace is missing.

    # 4a) Normal well-formed case: "answer": "..."
    m_ans = re.search(
        r'["\']answer["\']\s*:\s*["\']([^"\']*)["\']',
        s_clean,
    )

    # 4b) Fallback: allow missing closing quote/brace; grab to end of string
    if not m_ans:
        m_ans = re.search(
            r'["\']answer["\']\s*:\s*["\']?(.*)$',
            s_clean,
            flags=re.DOTALL,
        )

    if m_ans:
        ans = m_ans.group(1).strip()
        # Strip common trailing garbage like unmatched braces/commas/quotes
        ans = re.sub(r'[\s}"\']+$', "", ans)
        if ans:
            out["answer"] = ans

    return out if out else None



def extract_answer(
    raw_answer: str,
) -> Tuple[str, bool]:
    """
    Extract the "answer" string from the model's JSON response.
    Returns (answer, success_flag).
    """
    def _fail() -> Tuple[str, bool]:
        return "", False

    data = _load_json_dict(raw_answer)
    if data is None:
        print("[extract_answer] failed to parse JSON")
        print(raw_answer)
        return _fail()

    answer_value = data.get("answer")
    if not isinstance(answer_value, str):
        print("[extract_answer] 'answer' key missing or not a string")
        return _fail()

    return answer_value, True


def extract_citations(
    raw_answer: str,
    retrieved_docs: List[Dict[str, Any]],
) -> Tuple[List[str], bool]:
    """
    Extract citation doc_ids and answerable flag from the model's JSON response.
    Returns (cited_doc_ids, success_flag).
    """
    def _fail() -> Tuple[str, bool]:
        return [], False

    data = _load_json_dict(raw_answer)
    if data is None:
        print("[extract_citations] failed to parse JSON")
        print(raw_answer)
        return _fail()

    raw_citations = data.get("citations")
    if not isinstance(raw_citations, list):
        print("[extract_citations] 'citations' key missing or not a list")
        return _fail()

    indices: List[int] = []
    for c in raw_citations:
        try:
            indices.append(int(c))
        except Exception:
            print(f"[extract_citations] invalid citation index: {c}")
            return  _fail()

    cited_doc_ids = map_citation_indices_to_doc_ids(indices, retrieved_docs)

    return cited_doc_ids, True


def extract_answerable(
    raw_answer: str,
) -> Tuple[bool, bool]:
    """
    Extract the "answerable" boolean from the model's JSON response.
    Returns (answerable_flag, success_flag).
    """
    def _fail() -> Tuple[bool, bool]:
        # (answerable, success_flag)
        return False, False

    data = _load_json_dict(raw_answer)
    if data is None:
        print("[extract_answerable] failed to parse JSON")
        print(raw_answer)
        return _fail()

    answerable_value = data.get("answerable")
    if not isinstance(answerable_value, bool):
        print("[extract_answerable] 'answerable' key missing or not a boolean")
        return _fail()

    return answerable_value, True


def parse_gold_answers(raw: Any) -> str:
    """
    Extract a single gold answer string from a SQuAD-like blob:
      {"text": ["answer"], "answer_start": [...]}
    Returns an empty string when missing.
    """
    obj = raw
    if isinstance(raw, str):
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            obj = raw

    if isinstance(obj, dict):
        texts = obj.get("text")
        if isinstance(texts, list):
            return str(texts[0]) if texts else ""
        if isinstance(texts, str):
            return texts
        return ""

    if isinstance(obj, (list, tuple)) and obj:
        return str(obj[0])

    return str(obj) if obj else ""


def build_judge_prompt(
    question: str,
    model_answer: str,
    gold_answer: str,
) -> str:
    """
    Build a JSON-only grading prompt for a single gold answer.
    The judge should respond with: {"correct": True} or {"correct": False}
    """
    gold_str = gold_answer or ""
    return (
        "You are a strict QA grader. Decide if the model answer correctly answers the question.\n\n"
        'Return JSON ONLY with this shape (no extra text): {"correct": true|false}\n\n'
        "Rules:\n"
        "- Mark correct if the model answer semantically matches the gold answer.\n"
        f"Question: {question}\n"
        f"Model answer: {model_answer}\n"
        f"Gold answer: {gold_str}\n\n"
        "JSON Response:"
    )


def judge_answer(
    judge_client: Any,
    question: str,
    model_answer: str,
    gold_answer: str,
) -> bool:
    """
    Call an external judge model to decide correctness.

    judge_client is expected to have a .generate(prompt=..., **kwargs) method
    that returns a JSON string or an object with a .text / .content attribute.
    """
    prompt = build_judge_prompt(question, model_answer, gold_answer)

    try:
        resp = judge_client.generate(prompt=prompt, max_tokens=64)

        if isinstance(resp, str):
            text = resp
        elif hasattr(resp, "text"):
            text = resp.text
        elif hasattr(resp, "content"):
            text = resp.content
        elif hasattr(resp, "choices") and resp.choices:
            choice = resp.choices[0]
            if hasattr(choice, "message"):
                text = choice.message["content"]
            else:
                text = str(choice)
        else:
            text = str(resp)

        # Strip common code fences / whitespace
        text = text.strip().strip("`").strip()

        json_sub = _extract_json_substring(text)
        if json_sub is None:
            print(f"[judge_error] no JSON found in judge response: {text[:200]}")
            return False
        try:
            parsed = json.loads(json_sub)
            return bool(parsed.get("correct", False))
        except Exception as e:
            # Retry with a lenient parser to handle True/False or single quotes
            try:
                parsed = ast.literal_eval(json_sub)
                if isinstance(parsed, dict):
                    return bool(parsed.get("correct", False))
            except Exception:
                pass
            print(f"[judge_error] {e} raw: {text[:200]}")
            return False
    except Exception as e:
        print(f"[judge_error] {e}")
        return False

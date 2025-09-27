from typing import Iterable, List, Optional
import os


class ResponseGenerator:
    """
    Generation utility with two modes:
    - Groq (if GROQ_API_KEY is set): streams responses from Chat Completions.
    - Heuristic fallback: concatenates contexts into a pseudo-answer.
    """

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.model = model
        self._client = None
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq  # type: ignore

                self._client = Groq(api_key=api_key)
            except Exception:
                self._client = None

    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        context_blob = "\n\n".join(contexts[:5]) if contexts else ""
        return (
            "You are a helpful assistant that answers strictly based on the provided context.\n"
            "Cite or quote relevant snippets. If the answer cannot be found, say you don't know.\n\n"
            f"Context:\n{context_blob}\n\n"
            f"Question: {query}\n"
        )

    def generate_answer(self, query: str, contexts: List[str], stream: bool = True) -> Iterable[str] | str:
        """
        If streaming and Groq is available, yields chunks of text; otherwise returns a full string.
        """
        if not contexts and not self._client:
            return "I could not find relevant context. Please add data and rebuild the index."

        prompt = self._build_prompt(query, contexts)

        if self._client is None:
            # Heuristic fallback (non-streaming)
            return (
                "Heuristic answer based on retrieved context.\n\n"
                f"Question: {query}\n\n"
                f"Context used:\n{prompt}\n\n"
                "(Set GROQ_API_KEY to enable real LLM streaming.)"
            )

        if stream:
            # Stream tokens from Groq (OpenAI-compatible schema)
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You answer based only on the given context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                stream=True,
            )
            def _gen():
                for chunk in resp:
                    try:
                        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                    except Exception:
                        delta = None
                    if delta:
                        yield delta
            return _gen()
        else:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You answer based only on the given context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            try:
                return resp.choices[0].message.content or ""
            except Exception:
                return ""

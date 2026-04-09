import re
from dataclasses import dataclass, field


@dataclass
class CitationResult:
    answer: str
    is_grounded: bool
    citations_found: list = field(default_factory=list)
    confidence: float = 0.0


class CitationChecker:
    """
    Verify that the LLM answer is grounded in the retrieved context.

    Strategy:
    1. Extract citation markers like [1], [2] from the answer.
    2. For each non-trivial sentence in the answer, check whether any 4-gram
       from it appears in the retrieved chunks. If yes, that sentence is
       considered grounded.
    3. Confidence = grounded_sentences / total_sentences.
    4. If confidence < threshold, flag the answer as potentially hallucinated.
    """

    def __init__(self, overlap_threshold: float = 0.3):
        self.overlap_threshold = overlap_threshold

    def check(self, answer: str, retrieved_chunks: list) -> CitationResult:
        # 1. Extract citation markers
        citations = re.findall(r'\[\d+\]', answer)

        # 2. Check n-gram overlap between answer and context
        answer_sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
        context_text = " ".join(retrieved_chunks).lower()

        grounded_count = 0
        for sentence in answer_sentences:
            words = sentence.lower().split()
            if len(words) < 4:
                continue
            for i in range(len(words) - 3):
                ngram = " ".join(words[i:i+4])
                if ngram in context_text:
                    grounded_count += 1
                    break

        total = max(len(answer_sentences), 1)
        confidence = grounded_count / total

        return CitationResult(
            answer=answer,
            is_grounded=confidence >= self.overlap_threshold,
            citations_found=citations,
            confidence=confidence,
        )

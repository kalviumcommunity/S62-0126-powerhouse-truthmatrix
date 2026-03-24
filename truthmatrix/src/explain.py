from __future__ import annotations

import re
from typing import List


SENSATIONAL_WORDS = {
	"shocking",
	"breaking",
	"exclusive",
	"unbelievable",
	"urgent",
	"must-see",
	"exposed",
	"bombshell",
	"scandal",
	"outrageous",
}


def generate_explanation(text: str) -> List[str]:
	"""Generate simple human-readable reasons based on text characteristics."""
	content = (text or "").strip()
	explanations: List[str] = []

	# Rule 1: Article length
	if len(content) < 120:
		explanations.append("Article is very short")

	# Rule 2: Uppercase ratio
	uppercase_ratio = (
		sum(1 for ch in content if ch.isupper()) / len(content) if len(content) > 0 else 0.0
	)
	if uppercase_ratio > 0.12:
		explanations.append("Contains excessive capital letters")

	# Rule 3: Sensational word detection
	words = set(re.findall(r"[a-zA-Z-]+", content.lower()))
	matched_words = sorted(words.intersection(SENSATIONAL_WORDS))
	if matched_words:
		explanations.append(
			"Contains sensational words: " + ", ".join(matched_words)
		)

	if not explanations:
		explanations.append("No major red flags detected based on simple checks")

	return explanations


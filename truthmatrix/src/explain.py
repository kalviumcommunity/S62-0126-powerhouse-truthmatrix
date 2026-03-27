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

POSITIVE_WORDS = {
	"amazing",
	"excellent",
	"great",
	"good",
	"win",
	"success",
	"benefit",
	"safe",
	"trusted",
}

NEGATIVE_WORDS = {
	"terrible",
	"horrible",
	"bad",
	"fraud",
	"scam",
	"danger",
	"fear",
	"crisis",
	"fake",
}


def _normalize_label(label: str) -> str:
	lower_label = str(label or "").strip().lower()
	if lower_label in {"fake", "false", "0"}:
		return "Fake"
	if lower_label in {"real", "true", "1"}:
		return "Real"
	return "Unknown"


def generate_explanation(text: str, prediction_label: str) -> List[str]:
	"""Generate reasons for why content is classified as fake or real."""
	content = (text or "").strip()
	label = _normalize_label(prediction_label)
	explanations: List[str] = []

	if not content:
		return ["No text content available, so explanation is based on limited evidence."]

	tokens = re.findall(r"[a-zA-Z-]+", content.lower())
	word_set = set(tokens)

	# Rule 1: Article length
	if len(content) < 120:
		explanations.append("Text is short, which reduces context and increases uncertainty.")

	# Rule 2: Uppercase/exclamation style cues
	uppercase_ratio = (
		sum(1 for ch in content if ch.isupper()) / len(content) if len(content) > 0 else 0.0
	)
	exclamation_count = content.count("!")
	if uppercase_ratio > 0.12:
		explanations.append("Contains excessive capital letters, a common attention-grabbing pattern.")
	if exclamation_count >= 2:
		explanations.append("Uses multiple exclamation marks, which can indicate clickbait framing.")

	# Rule 3: Clickbait word detection
	matched_words = sorted(word_set.intersection(SENSATIONAL_WORDS))
	if matched_words:
		explanations.append("Contains clickbait words: " + ", ".join(matched_words))

	# Rule 4: Sentiment analysis hints (lexicon-based)
	positive_hits = len(word_set.intersection(POSITIVE_WORDS))
	negative_hits = len(word_set.intersection(NEGATIVE_WORDS))
	sentiment_score = positive_hits - negative_hits

	if sentiment_score >= 2:
		explanations.append("Tone appears strongly positive, which may reflect persuasive framing.")
	elif sentiment_score <= -2:
		explanations.append("Tone appears strongly negative, which may reflect fear-based framing.")
	else:
		explanations.append("Tone appears relatively neutral based on sentiment cues.")

	# Rule 5: Mismatch with expected patterns for predicted class
	if label == "Fake":
		if not matched_words and exclamation_count == 0 and uppercase_ratio < 0.08 and abs(sentiment_score) <= 1:
			explanations.append(
				"Mismatch detected: the writing style looks neutral/structured for a fake classification."
			)
		else:
			explanations.append(
				"Pattern match: sensational or emotional cues support the fake classification."
			)
	elif label == "Real":
		if matched_words or exclamation_count >= 2 or uppercase_ratio > 0.12 or abs(sentiment_score) >= 2:
			explanations.append(
				"Mismatch detected: emotional/clickbait style is unusual for a real classification."
			)
		else:
			explanations.append(
				"Pattern match: balanced tone and low clickbait cues align with real classification."
			)
	else:
		explanations.append("Prediction label is unknown, so pattern matching is limited.")

	if not explanations:
		explanations.append("No major textual cues detected based on rule-based checks.")

	return explanations


def calculate_credibility_score(
	text: str,
	text_confidence: float = 0.5,
	image_confidence: float = 0.5,
	fusion_confidence: float = 0.5,
) -> float:
	"""Calculate credibility score (0-100) based on text analysis and model confidence.
	
	Factors:
	- Sentiment polarity (neutral is more credible)
	- Presence of exaggerated/sensational words (fewer is better)
	- Model prediction confidence (higher is more credible)
	- Text structure and length (well-structured is more credible)
	
	Args:
		text: The text content to analyze
		text_confidence: Text model confidence [0, 1]
		image_confidence: Image model confidence [0, 1]
		fusion_confidence: Fusion model confidence [0, 1]
	
	Returns:
		credibility_score: Score from 0 to 100
	"""
	content = (text or "").strip()
	
	if not content:
		avg_confidence = (text_confidence + image_confidence + fusion_confidence) / 3.0
		return max(0.0, min(100.0, avg_confidence * 100.0))
	
	tokens = re.findall(r"[a-zA-Z-]+", content.lower())
	word_set = set(tokens)
	total_words = len(tokens)
	
	score = 75.0
	
	# Factor 1: Sentiment polarity (neutral = more credible)
	positive_hits = len(word_set.intersection(POSITIVE_WORDS))
	negative_hits = len(word_set.intersection(NEGATIVE_WORDS))
	sentiment_score = positive_hits - negative_hits
	
	if abs(sentiment_score) <= 1:
		pass
	elif abs(sentiment_score) <= 3:
		score -= 5.0
	else:
		score -= 10.0
	
	# Factor 2: Exaggerated/sensational words (fewer is better)
	sensational_count = len(word_set.intersection(SENSATIONAL_WORDS))
	if sensational_count > 0:
		sensational_ratio = sensational_count / max(1, total_words)
		if sensational_ratio > 0.05:
			score -= 15.0
		elif sensational_ratio > 0.02:
			score -= 8.0
		else:
			score -= 3.0
	
	# Factor 3: Text structure (short text is less credible)
	if len(content) < 50:
		score -= 15.0
	elif len(content) < 120:
		score -= 8.0
	
	# Factor 4: Excessive capitalization
	uppercase_ratio = sum(1 for ch in content if ch.isupper()) / len(content) if len(content) > 0 else 0.0
	if uppercase_ratio > 0.12:
		score -= 10.0
	elif uppercase_ratio > 0.08:
		score -= 5.0
	
	# Factor 5: Punctuation patterns (excessive exclamation/question marks)
	exclamation_count = content.count("!")
	question_count = content.count("?")
	excessive_punct = exclamation_count + question_count
	if excessive_punct >= 3:
		score -= 12.0
	elif excessive_punct >= 1:
		score -= 5.0
	
	# Factor 6: Model confidence (higher confidence = more credible)
	avg_confidence = (text_confidence + image_confidence + fusion_confidence) / 3.0
	confidence_boost = avg_confidence * 15.0
	score += confidence_boost
	
	credibility_score = max(0.0, min(100.0, score))
	return float(credibility_score)


# Fake Question Bank Dataset (for Artifact Review)

This dataset is **synthetic** but is formatted to look like a question-bank + student interaction log
collected in a programming education setting.

Files:
- `problems.jsonl`: question bank (problem_id, topic, difficulty, text, code, concept_hints)
- `interactions.jsonl`: student attempts (user_id, problem_id, correct, timestamp, duration_sec, attempt, language)
- `user_profiles.jsonl`: student profiles (user_id, features)

Notes:
- Timestamps fall in a fictitious semester window (Sep–Dec 2025).
- Correctness is generated with a simple learning dynamic: topic-specific skill increases with practice.

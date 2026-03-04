EXTRACT_PROMPT = """Extract programming concepts from the following solution code and/or problem statement.

Return STRICT JSON with keys:
- data_structures: list[str]
- algorithms: list[str]
- paradigms: list[str]
- time_complexity: str (optional)
- space_complexity: str (optional)

Input:
[PROBLEM_TEXT]
{problem_text}

[SOLUTION_CODE]
```python
{code}
```
"""

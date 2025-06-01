def build_prompt(requirement_text: str, self) -> str:
    return f"""
You are an expert Agile Business Analyst.
Given the following raw requirement:
\"{requirement_text}\"
Generate a single INVEST-compliant user story in this format:
As a [user type], I want to [goal], so that [benefit].
DO NOT provide explanations or multiple options.
"""
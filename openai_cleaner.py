import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_clean_and_validate(raw_dict):
    prompt = f"""
    You are a data cleaner for a Neo4j system. Standardize this input dictionary:
    - Ensure keys are snake_case.
    - Detect malformed/mislabeled fields.
    - Return valid fields only with inferred fixes.

    Input:
    {raw_dict}

    Output as valid JSON:
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

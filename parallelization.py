import asyncio
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# -----------------------------
# LLM setup
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -----------------------------
# Async extraction agents
# -----------------------------
async def extract_cities(query):
    prompt = PromptTemplate.from_template(
        """
        Extract city names from the query below.
        Return JSON in the form:
        {{"cities": ["city1", "city2"]}}

        Query:
        {query}
        """
    )
    return await llm.ainvoke(prompt.format(query=query))


async def extract_amenities(query):
    prompt = PromptTemplate.from_template(
        """
        Extract amenity changes with numeric values from the query.
        Return JSON in the form:
        {{"amenity_changes": {{"amenity": value}}}}

        Query:
        {query}
        """
    )
    return await llm.ainvoke(prompt.format(query=query))


async def classify_intent(query):
    prompt = PromptTemplate.from_template(
        """
        Classify the intent of the query as one of:
        prediction | explanation | comparison

        Return ONLY the label.

        Query:
        {query}
        """
    )
    return await llm.ainvoke(prompt.format(query=query))

# -----------------------------
# Robust JSON parser
# -----------------------------
def clean_json(text: str):
    """
    Extract the first JSON object from LLM output and parse it.
    """
    if not text or not text.strip():
        raise ValueError("Empty response from LLM")

    # Remove markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response:\n{text}")

    return json.loads(match.group())

# -----------------------------
# Parallel parser
# -----------------------------
async def parallel_parse(query):
    results = await asyncio.gather(
        extract_cities(query),
        extract_amenities(query),
        classify_intent(query),
    )

    parsed = {
        "cities": clean_json(results[0].content)["cities"],
        "amenities": clean_json(results[1].content)["amenity_changes"],
        "intent": results[2].content.strip().lower()
    }

    return parsed

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    query = "If Bangalore improves transport by 10% and Delhi loses parks by 5%"
    parsed = asyncio.run(parallel_parse(query))
    print(parsed)

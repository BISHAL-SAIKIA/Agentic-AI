from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
#Defining tool
def get_population(city: str) -> int:
    data = {
        "Delhi": 33_000_000,
        "Mumbai":21_000_000,
        "Bangalore":13_000_000
    }
    return data.get(city, 0)

# Describe the tool to the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get population of an Indian city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

client = OpenAI()

response = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages=[{"role":"user","content":"Population of Delhi"}],
    tools = tools,
    tool_choice = "auto"
)

#Detect tool call

tool_call = response.choices[0].message.tool_calls[0]
args = tool_call.function.arguments

result = get_population(**eval(args))

#Feed tool result back
messages = [
    {"role": "user", "content": "Population of Delhi?"},
    response.choices[0].message,
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    }
]

final = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages
)

print(final.choices[0].message.content)

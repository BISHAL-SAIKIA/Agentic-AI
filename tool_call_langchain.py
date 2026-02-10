from openai import OpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
load_dotenv()

@tool
def get_population(city:str) ->int:
    """ Get population of an Indian city """
    data = {
        "Delhi":33_000_000,
        "Mumbai":21_000_000,
        "Bangalore":13_000_000
    }

    return data.get(city,0)

# Creating the llm with tools bound
llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature=0
)

tools = [get_population]
llm_with_tools = llm.bind_tools(tools)

query = HumanMessage(content = "What is the population of Delhi")
response = llm_with_tools.invoke([query])

# print(response.tool_calls)


tool_call = response.tool_calls[0]

tool_result = get_population.invoke(tool_call["args"])


tool_message = ToolMessage(
    content=str(tool_result),
    tool_call_id=tool_call["id"]
)

final_response = llm.invoke([
    query,
    response,
    tool_message
])

print(final_response.content)

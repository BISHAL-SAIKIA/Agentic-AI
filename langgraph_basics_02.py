from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()


class State(TypedDict):
    question: str
    cleaned_question: str
    raw_answer: str
    final_answer:str


# Initialize OpneAI LLM

llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature = 0.2
)

# Node 1 : Clean the input
def clean_input(state: State):
    """
    Simple preprocessing:
    - strip extra spaces
    - make text lowercase
    """

    cleaned = state["question"].strip().lower()
    return {"cleaned_question":cleaned}

# Node 2 : Call OpenAI LLM
def llm_call(state: State):
    """
    Sends cleaned question to LLM
    """
    response = llm.invoke(state["cleaned_question"])
    return {"raw_answer": response.content}

# Node 3 :  Format the answer 
def format_answer(state: State):
    """
    Adds a label to the LLM output
    """
    formatted = f"Answer:\n{state['raw_answer']}"
    return {"final_answer":formatted}

# Building the graph
graph = StateGraph(State)

graph.add_node("clean_input", clean_input)
graph.add_node("llm_call",llm_call)
graph.add_node("format_answer",format_answer)

graph.set_entry_point("clean_input")
graph.add_edge("clean_input","llm_call")
graph.add_edge("llm_call","format_answer")
graph.add_edge("format_answer",END)

# Compile graph
app = graph.compile()

# Run

if __name__ =="__main__":
    user_question = "Tell me about the state Raipur, India"
    result = app.invoke({"question": user_question})
    print(result["final_answer"])
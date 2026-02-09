from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# State = what flows between nodes
class ChatState(TypedDict):
    question: str
    answer: str


#Creating a node

llm = ChatOpenAI(model="gpt-3.5-turbo")

def answer_node(state:ChatState):
    response = llm.invoke(state["question"])
    return {"answer": response.content}



#Building the graph
graph = StateGraph(ChatState)

graph.add_node("answer", answer_node)

graph.set_entry_point("answer")
graph.add_edge("answer",END)

app = graph.compile()


result = app.invoke({"question": "Tell me about lang graph framework used to make Agents"})
print(result["answer"])
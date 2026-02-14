from typing import TypedDict

from langgraph.graph import END, StateGraph


class RAGState(TypedDict):
    # Minimal state placeholder for the LangGraph workflow.
    query: str
    classification: str
    documents: list
    answer: str
    verification: dict


def classify_node(state: RAGState) -> RAGState:
    # TODO: implement classification logic.
    return state


def retrieve_node(state: RAGState) -> RAGState:
    # TODO: implement retrieval logic.
    return state


def generate_node(state: RAGState) -> RAGState:
    # TODO: implement generation logic.
    return state


def verify_node(state: RAGState) -> RAGState:
    # TODO: implement verification logic.
    return state


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()

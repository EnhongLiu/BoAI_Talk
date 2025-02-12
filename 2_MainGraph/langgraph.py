from typing import TypedDict, Sequence
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langchain.schema import BaseMessage
from PIL import Image  # If `graph.get_graph(xray=True).draw_mermaid_png()` returns an image


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes before inference
workflow.add_node("decide_model_among_tools", decide_model)
workflow.add_node("extract_key_parameters", prepare_query_for_milk_yield_visuals)
workflow.add_node("generate_visuals", milkbot_visuals)
workflow.add_node("rewrite_questions", rewrite)

# Connect agent nodes
workflow.add_edge(START, "decide_model_among_tools")
workflow.add_conditional_edges("decide_model_among_tools", tools_condition, {"tools": "extract_key_parameters", END:END})
workflow.add_conditional_edges("extract_key_parameters", grade_answer, {"milkbot_visuals": 'generate_visuals', "rewrite": 'rewrite_questions'})
workflow.add_edge('rewrite_questions', 'decide_model_among_tools')
workflow.add_edge('generate_visuals', END)

# Compile
graph = workflow.compile()

# Display graph
try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

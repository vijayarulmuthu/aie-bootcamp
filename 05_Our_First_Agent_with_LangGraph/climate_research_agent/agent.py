import os
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_community.tools import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


# Define state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Load tools
tool_belt = [
    TavilySearchResults(max_results=5),
    ArxivQueryRun(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
]

# Bind model
llm = ChatOpenAI(model="gpt-4.1", temperature=0).bind_tools(tool_belt)

def call_model(state):
    return {"messages": [llm.invoke(state["messages"])]}

# Helpfulness check
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

helpfulness_model = ChatOpenAI(model="gpt-4", temperature=0)
prompt_template = PromptTemplate.from_template("""
Given a user's initial query and the agent's final response, answer with 'Y' if the response is helpful and complete, or 'N' otherwise.  Make sure you don't return JSON output.

Initial Query:
{initial_query}

Final Response:
{final_response}
""")
helpfulness_chain = prompt_template | helpfulness_model | StrOutputParser()

def tool_or_helpful(state):
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "action"
    if len(state["messages"]) > 10:
        return "end"
    initial = state["messages"][0].content
    final = last_msg.content
    result = helpfulness_chain.invoke({
        "initial_query": initial,
        "final_response": final
    })
    return "end" if "Y" in result else "continue"

tool_node = ToolNode(tool_belt)

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("action", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tool_or_helpful, {
    "action": "action",
    "continue": "agent",
    "end": END
})
graph.add_edge("action", "agent")
climate_agent = graph.compile()

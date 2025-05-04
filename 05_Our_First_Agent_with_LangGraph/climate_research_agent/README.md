# Climate Change Research Agent

**Problem Statement:**
Busy sustainability researchers and climate policy analysts struggle to synthesize credible scientific, governmental, and current-event information in real time.

## User Persona & Problem Context
The target user is a **climate researcher**, **sustainability consultant**, or **policy analyst** working in government, NGOs, or research institutes. Their role demands constant awareness of the latest scientific studies, policy developments, and global trends related to climate change and environmental sustainability.

These users frequently ask questions like:

* “What are the latest research findings on permafrost melting?”
* “Which countries are leading renewable energy adoption in 2024?”
* “How do IPCC projections compare with recent sea-level data?”

The problem is that their current research workflow is fragmented — requiring them to:

* Search academic databases (e.g., Arxiv, ScienceDirect)
* Monitor policy reports (e.g., UN, IPCC, EPA)
* Track relevant news articles and blogs

This manual process is **time-consuming**, often involves switching between platforms, and leads to **information overload** or **missed updates**. Automating this synthesis — through an intelligent agent that can query multiple sources, loop over tools, and deliver summarized insights — would significantly improve their productivity and confidence in their decisions.


## Propose a Solution
We propose an AI-powered agent that intelligently orchestrates multiple tools (Arxiv, Wikipedia, Tavily) using LangGraph to create a multi-turn, tool-aware assistant.

Tools:

* ArxivQueryRun – Academic climate research
* WikipediaQueryRun – Contextual/encyclopedic data
* TavilySearchResults – Recent climate news and policy developments

The agent cycles through tool calls and uses a helpfulness check to ensure the response is informative before delivering it to the user via Chainlit.


## Dealing with the Data
The agent uses:

* RAG with LangChain and tools (no static corpus required)
* External APIs: Arxiv, Tavily, and Wikipedia

Data is sourced dynamically at runtime from public APIs. The helpfulness check ensures only complete, helpful responses are delivered.


## Setup

```bash
uv sync
source .venv/bin/activate
chainlit run chainlit_app.py -w
```

## Example Questions

- What is the latest research on ocean acidification?
- Who are the leading countries in renewable energy?
- How does permafrost melting affect global emissions?
- What role does ocean acidification play in marine biodiversity?
- Which countries lead in renewable energy adoption as of 2024?
- Summarize the latest IPCC reports on sea level rise.

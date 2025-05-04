import chainlit as cl
from langchain_core.messages import HumanMessage
from agent import climate_agent

@cl.on_message
async def handle_message(message: cl.Message):
    inputs = {"messages": [HumanMessage(content=message.content)]}

    final_response = None
    async for update in climate_agent.astream(inputs, stream_mode="updates"):
        for node, values in update.items():
            final_response = values["messages"][-1].content

    await cl.Message(content=final_response or "✅ Done.").send()
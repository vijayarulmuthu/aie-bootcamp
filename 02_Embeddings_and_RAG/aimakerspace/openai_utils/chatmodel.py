import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from .prompts import SystemRolePrompt, UserRolePrompt

load_dotenv()


class ChatOpenAI:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 1):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")

    async def generate_response_with_context(self, context: str, query: str) -> str:
        """Generate a response using the provided context and query"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, say 'I don't have enough information to answer this question.' Do not make up information."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        return await self.generate_response(messages)

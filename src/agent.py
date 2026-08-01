"""The agent loop: Claude autonomously calls finance tools to answer a question.

This is a hand-written agentic loop (not the SDK's tool runner) to make the
request -> tool_use -> execute -> feed-back -> repeat cycle explicit. Claude
decides which tools to call and when it has enough to write the final analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from . import config
from .tools import TOOL_SCHEMAS, run_tool

SYSTEM_PROMPT = """You are a financial analyst agent. Answer the user's question by \
calling the available tools to gather real market data and risk metrics, then \
synthesize a clear, well-structured analysis.

Guidelines:
- Call tools to get actual numbers; never invent figures.
- Chain multiple tools when a thorough answer needs it (e.g. snapshot + volatility + VaR).
- When you have enough data, write a concise analysis with the key numbers and a plain-English takeaway.
- You are not a licensed financial advisor; do not give personalized investment advice. \
Present analysis and let the reader decide."""


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)


def run_agent(question: str, verbose: bool = False) -> AgentResult:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    messages: list[dict] = [{"role": "user", "content": question}]
    tool_calls: list[dict] = []

    for _ in range(config.MAX_TURNS):
        response = client.messages.create(
            model=config.AGENT_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            # Claude is done — return the final text.
            answer = next((b.text for b in response.content if b.type == "text"), "")
            return AgentResult(answer=answer, tool_calls=tool_calls)

        # Record the assistant turn (including tool_use blocks) verbatim.
        messages.append({"role": "assistant", "content": response.content})

        # Execute each requested tool and collect results.
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block.name, block.input)
                tool_calls.append({"name": block.name, "input": block.input, "output": output})
                if verbose:
                    print(f"→ {block.name}({block.input})\n{output}\n")
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )

        messages.append({"role": "user", "content": results})

    return AgentResult(
        answer="Stopped: reached the maximum number of tool-calling turns.",
        tool_calls=tool_calls,
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Analyze the risk profile of NVDA."
    result = run_agent(q, verbose=True)
    print("=" * 60)
    print(result.answer)
    print("=" * 60)
    print(f"({len(result.tool_calls)} tool calls made)")

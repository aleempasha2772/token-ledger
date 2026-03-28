# llm-tracker

Track token usage and cost across LLM calls with one decorator.

## Install

pip install token-ledger


## Quick Start

from token_ledger import track_llm_cost, configure

configure(project_name="my_app", budget=5.00)

@track_llm_cost(model="gpt-4o")
def generate(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)

## What You Get

- Per-call cost logging
- Session aggregation
- Budget cap with BudgetExceededError
- Agent call tree (parent → child linking)
- File report on exit
- Zero hard dependencies

## Supported Providers

OpenAI, Anthropic, Google Gemini, Ollama, any OpenAI-compatible server

## Optional Dependencies

pip install llm-tracker[openai]      # OpenAI SDK
pip install llm-tracker[anthropic]   # Anthropic SDK
pip install llm-tracker[all]         # Everything
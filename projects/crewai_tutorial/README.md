# CrewAI Multi-Agent Tutorial

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![CrewAI](https://img.shields.io/badge/CrewAI-1.0+-orange)](https://crewai.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT%204-412991?logo=openai&logoColor=white)](https://openai.com)
[![Gemini](https://img.shields.io/badge/Gemini-Pro-4285F4?logo=google&logoColor=white)](https://deepmind.google/gemini)

A comprehensive, example-driven tutorial covering **CrewAI's core concepts** — from single agents to multi-agent collaboration, tool integration, hierarchical management, YAML-based configuration, and RAG.

---

## Features

| # | Example | Concepts |
|---|---------|----------|
| 1 | Simple Agent | Agent creation, task assignment |
| 2 | Multi-Agent Collaboration | Sequential pipeline (researcher → writer → editor) |
| 3 | Tools | Web search integration (SerperDev, DuckDuckGo) |
| 4 | Hierarchical Process | Manager agent delegates to specialists |
| 5 | Custom Tools | Build and register custom tools |
| 6 | Full Working Example | End-to-end runnable demo |
| 7 | File Output | Save agent results to files |
| 8 | Parallel Tasks | Independent concurrent execution |
| 9 | YAML Config | `@CrewBase` pattern with agents/tasks YAML |

---

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI or Gemini key

# Run the main tutorial
python crewai_tutorial.py

# Or run the YAML-based crew
python 2_yaml.py
```

---

## Project Structure

```
├── crewai_tutorial.py     # Main tutorial (8 examples, commented)
├── 2_yaml.py              # YAML-configured CrewAI crew
├── 1_mailAgent.ipynb      # Gemini-powered mail agent notebook
├── config/
│   ├── agents.yaml        # Agent definitions
│   └── tasks.yaml         # Task definitions
├── .env.example           # API key configuration template
└── requirements.txt       # Dependencies
```

---

## Key Concepts Demonstrated

- **Agents**: Role-based AI entities with specific goals and backstories
- **Tasks**: Work items with expected outputs, dependencies, and context
- **Tools**: Web search, custom tools, and RAG retrieval attached to agents
- **Crews**: Orchestration of agents in sequential, hierarchical, or parallel processes
- **YAML Config**: Clean separation of agent/task definitions from code
- **Multi-LLM**: Works with OpenAI GPT-4, Google Gemini, and other providers

---

## References

- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/crewAI/crewai)
- [LangChain](https://python.langchain.com/)

# AI/ML Coursework & Projects

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![CrewAI](https://img.shields.io/badge/CrewAI-1.0+-orange)](https://crewai.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-vector--db-5291D1)](https://github.com/facebookresearch/faiss)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?logo=openai&logoColor=white)](https://openai.com)

A comprehensive collection of AI/ML coursework, assignments, and personal projects — from foundational Python through advanced multi-agent RAG systems. Organized for both **portfolio showcase** and **revision/learning reference**.

---

## Structure

```
├── assignments/               ← Graded submissions (showcase + revision)
│   ├── 01_week04_mini_project/   Python + data analysis (library mgmt, retail insights)
│   ├── 02_week06_project/        Graded project
│   ├── 03_week12_graded_project/ Machine learning project
│   ├── 04_week15_rag_assistant/  RAG chatbot with guardrails, FAISS, Streamlit
│   └── 05_week18_mini_project/   Ethics, safety & governance report
├── projects/                  ← Complete personal projects (showcase)
│   ├── hr_rag_chatbot/           HR Support RAG chatbot (LangChain + FAISS + Streamlit)
│   ├── crewai_tutorial/          Multi-agent CrewAI tutorial (8 patterns, YAML config)
│   └── image_processing/         Transformer-based image embedding & similarity
├── reference_notebooks/       ← Learning materials (revision)
│   ├── langchain/                LangChain memory, agents, chunking strategies
│   ├── rag/                      RAG notebooks (IBM SkillsNetwork)
│   └── additional/               Chatbot creation, machine translation
└── reference_materials/       ← Course slides and cheat sheets
```

---

## Assignments

| Week | Project | Tech Stack |
|------|---------|------------|
| 4 | Mini Project — City Library + Retail Analysis | Python, Pandas, Seaborn |
| 6 | Graded Project | Data analysis |
| 12 | Graded Project — ML Pipeline | Python, ML |
| 15 | Tech Documentation RAG Assistant | LangChain, FAISS, Streamlit, Guardrails |
| 18 | Ethics, Safety & Governance | AI safety principles |

---

## Projects

### HR RAG Support Chatbot
RAG-based chatbot answering employee HR policy questions using official company documents. Features PDF ingestion, FAISS vector search, Streamlit UI, conversation memory, and LangSmith observability.
→ `projects/hr_rag_chatbot/`

### CrewAI Multi-Agent Tutorial
Comprehensive tutorial covering 8 CrewAI patterns: simple agents, multi-agent collaboration, tool integration, hierarchical management, custom tools, parallel tasks, YAML-based configuration, and RAG integration.
→ `projects/crewai_tutorial/`

### Image Embedding & Similarity
Transformer-based (CLIP) image embedding generation and similarity visualization using a 9-class dataset.
→ `projects/image_processing/`

---

## Reference Notebooks

Pre-organized course materials from the IITM AAIA program and IBM SkillsNetwork for quick revision:
- **LangChain**: Memory types, first agent, chunking strategies
- **RAG**: Prompt engineering, in-context learning, RAG with PyTorch and HuggingFace
- **Additional**: Chatbot creation, machine translation basics

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/gamerdevarsh/AI-ML-Coursework.git
cd AI-ML-Coursework

# Run any assignment or project
# Each has its own requirements.txt; install as needed
cd assignments/04_week15_rag_assistant
pip install -r requirements.txt
streamlit run app.py
```

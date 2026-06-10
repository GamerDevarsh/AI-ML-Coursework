"""
CrewAI Tutorial - Complete Example with Key Functionality
=========================================================

This file demonstrates the core concepts of CrewAI:
1. Agents - AI entities with specific roles and goals
2. Tasks - Work items assigned to agents
3. Tools - Capabilities agents can use
4. Crews - Orchestration of multiple agents working together
5. Processes - How agents collaborate (sequential, hierarchical)

Setup:
------
Create a .env file with your API keys:
    OPENAI_API_KEY=your_key_here
    # Or use other LLM providers

Run:
    python crewai_tutorial.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# EXAMPLE 1: Simple Agent with Task
# ============================================================
"""
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Define an LLM (OpenAI or compatible)
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create an Agent with a specific role
researcher = Agent(
    role="Research Analyst",
    goal="Find and summarize the latest AI trends",
    backstory="You are an expert research analyst with 10 years of experience.",
    llm=llm,
    verbose=True
)

# Create a Task for the agent
research_task = Task(
    description="Research the top 3 AI trends in 2026 and provide a summary",
    agent=researcher,
    expected_output="A concise summary of 3 AI trends with brief explanations"
)

# Create a Crew and add the agent and task
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

# Run the crew
result = crew.kickoff()
print(result)
"""

# ============================================================
# EXAMPLE 2: Multi-Agent Collaboration (Sequential)
# ============================================================
"""
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create multiple specialized agents
researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive information on a topic",
    backstory="Expert researcher skilled at finding accurate information.",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write clear and engaging content based on research",
    backstory="Professional writer with expertise in creating compelling narratives.",
    llm=llm,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Review and refine content for quality",
    backstory="Senior editor with keen eye for detail and quality.",
    llm=llm,
    verbose=True
)

# Create tasks - note writer depends on researcher, editor depends on writer
research_task = Task(
    description="Research the latest developments in quantum computing",
    agent=researcher,
    expected_output="Detailed notes on 5 key developments in quantum computing"
)

write_task = Task(
    description="Write a 500-word article about quantum computing developments",
    agent=writer,
    expected_output="A well-structured article in markdown format",
    context=[research_task]  # writer waits for researcher to complete
)

edit_task = Task(
    description="Edit the article for clarity and flow",
    agent=editor,
    expected_output="Final polished article",
    context=[write_task]  # editor waits for writer to complete
)

# Create crew with sequential process (one after another)
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process="sequential",  # Default - tasks run in order
    verbose=True
)

result = crew.kickoff()
print(result)
"""

# ============================================================
# EXAMPLE 3: Using Tools (Web Search, RAG, etc.)
# ============================================================
"""
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, DirectDriver
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Initialize tools
search_tool = SerperDevTool()  # Web search
# Or for DuckDuckGo:
# from crewai_tools import DuckDuckGoSearchRun
# search_tool = DuckDuckGoSearchRun()

# Create agent with tools
researcher = Agent(
    role="Web Researcher",
    goal="Find accurate and up-to-date information from the web",
    backstory="Expert researcher who knows how to find reliable information online.",
    llm=llm,
    tools=[search_tool],  # Attach tools to the agent
    verbose=True
)

# Create a task that requires web search
research_task = Task(
    description="Find the latest news about SpaceX missions in 2026",
    agent=researcher,
    expected_output="Summary of 5 recent SpaceX news items with sources"
)

crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

result = crew.kickoff()
"""

# ============================================================
# EXAMPLE 4: Hierarchical Process (Manager Agent)
# ============================================================
"""
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create specialist agents
writer = Agent(
    role="Technical Writer",
    goal="Write clear technical documentation",
    backstory="Expert technical writer for software documentation.",
    llm=llm
)

coder = Agent(
    role="Code Reviewer",
    goal="Review code for quality and bugs",
    backstory="Senior software engineer with expertise in code review.",
    llm=llm
)

# Create a manager agent that delegates work
manager = Agent(
    role="Project Manager",
    goal="Coordinate team to deliver high-quality software",
    backstory="Experienced project manager who orchestrates complex projects.",
    llm=llm
)

write_task = Task(
    description="Write documentation for a new API endpoint",
    agent=writer
)

review_task = Task(
    description="Review the code changes for the API endpoint",
    agent=coder
)

# Use hierarchical process - manager delegates to agents
crew = Crew(
    agents=[manager, writer, coder],
    tasks=[write_task, review_task],
    process="hierarchical",  # Manager assigns tasks to agents
    manager_agent=manager,
    verbose=True
)

result = crew.kickoff()
"""

# ============================================================
# EXAMPLE 5: Custom Tools with crewai-tools
# ============================================================
"""
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import Field

# Create a custom tool
class WeatherTool(BaseTool):
    name: str = Field(default="weather_checker")
    description: str = Field(default="Get weather information for a city")
    
    def _run(self, city: str) -> str:
        # Simulated weather lookup (replace with real API)
        return f"Weather in {city}: Sunny, 72°F"

weather_tool = WeatherTool()

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

weather_agent = Agent(
    role="Weather Assistant",
    goal="Provide accurate weather information",
    backstory="Weather expert with access to global weather data.",
    llm=llm,
    tools=[weather_tool],
    verbose=True
)

weather_task = Task(
    description="Get the weather for New York and Paris",
    agent=weather_agent,
    expected_output="Weather summary for both cities"
)

crew = Crew(
    agents=[weather_agent],
    tasks=[weather_task],
    verbose=True
)

result = crew.kickoff()
"""

# ============================================================
# EXAMPLE 6: Full Working Example (Run This!)
# ============================================================
"""
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# Configure LLM - Set OPENAI_API_KEY in .env file
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create a simple researcher agent
researcher = Agent(
    role="AI Research Analyst",
    goal="Provide accurate and concise information about AI topics",
    backstory="""You are an experienced AI researcher who stays updated with 
    the latest developments in artificial intelligence. You provide clear, 
    factual summaries of complex topics.""",
    llm=llm,
    verbose=True
)

# Create a task
research_task = Task(
    description="""Research and summarize the key benefits of using multi-agent 
    systems in AI applications. Include at least 3 specific benefits with 
    brief explanations.""",
    agent=researcher,
    expected_output="A structured summary with 3-4 key benefits of multi-agent AI systems"
)

# Create crew and run
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=2  # Enable detailed logging
)

print("🚀 Starting CrewAI Task...")
result = crew.kickoff()
print("\n📋 Final Result:")
print(result)
"""

# ============================================================
# EXAMPLE 7: Creating Output Files
# ============================================================
"""
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create agent with file output capability
writer = Agent(
    role="Report Writer",
    goal="Write comprehensive reports",
    backstory="Professional report writer who creates detailed, well-structured documents.",
    llm=llm,
    verbose=True
)

# Task with output file
report_task = Task(
    description="Write a 300-word executive summary about AI in healthcare",
    agent=writer,
    expected_output="Professional report in markdown format",
    output_file="ai_healthcare_report.md"  # Save output to file
)

crew = Crew(
    agents=[writer],
    tasks=[report_task],
    verbose=True
)

result = crew.kickoff()
# Check ai_healthcare_report.md for the output
"""

# ============================================================
# EXAMPLE 8: Parallel Task Execution
# ============================================================
"""
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create agents for different topics
tech_agent = Agent(
    role="Tech Analyst",
    goal="Analyze technology trends",
    backstory="Technology expert with deep industry knowledge.",
    llm=llm
)

finance_agent = Agent(
    role="Finance Analyst",
    goal="Analyze financial trends",
    backstory="Financial expert with 15 years of market analysis experience.",
    llm=llm
)

health_agent = Agent(
    role="Health Analyst",
    goal="Analyze health industry trends",
    backstory="Healthcare industry expert.",
    llm=llm
)

# Create independent tasks (can run in parallel)
tech_task = Task(
    description="Summarize key trends in the tech industry for 2026",
    agent=tech_agent,
    expected_output="3-4 key trends with brief analysis"
)

finance_task = Task(
    description="Summarize key trends in the finance industry for 2026",
    agent=finance_agent,
    expected_output="3-4 key trends with brief analysis"
)

health_task = Task(
    description="Summarize key trends in the health industry for 2026",
    agent=health_agent,
    expected_output="3-4 key trends with brief analysis"
)

# Crew with parallel execution (when used with kickoff for independent tasks)
crew = Crew(
    agents=[tech_agent, finance_agent, health_agent],
    tasks=[tech_task, finance_task, health_task],
    verbose=True
)

# All tasks can potentially run in parallel if they don't depend on each other
result = crew.kickoff()
"""

# ============================================================
# RUNNING THE EXAMPLES
# ============================================================
"""
To run any example:
1. Uncomment the code block
2. Ensure OPENAI_API_KEY is set in .env
3. Run: python crewai_tutorial.py

For more examples and documentation, visit:
- https://docs.crewai.com/
- https://github.com/crewAI/crewai
"""

if __name__ == "__main__":
    print("""
    ================================================
    CrewAI Tutorial - Examples Ready to Run!
    ================================================
    
    This file contains multiple examples demonstrating:
    
    1. Simple Agent with Task - Basic agent setup
    2. Multi-Agent Collaboration - Sequential workflow
    3. Using Tools - Web search integration
    4. Hierarchical Process - Manager-led delegation
    5. Custom Tools - Building your own tools
    6. Full Working Example - Complete runnable code
    7. Output Files - Saving results to files
    8. Parallel Tasks - Independent task execution
    
    To run an example:
    1. Uncomment the example code block
    2. Set your OPENAI_API_KEY in .env
    3. Run: python crewai_tutorial.py
    
    ================================================
    """)
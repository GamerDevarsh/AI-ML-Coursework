from crewai import Agent, Crew, Process, Task, LLM 
from crewai.project import CrewBase, agent, crew, task # pyright: ignore[reportMissingImports]

from crewai_tools import SerperDevTool, ScrapeWebsiteTool, DirectoryReadTool # pyright: ignore[reportMissingImports]

from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
load_dotenv()

@CrewBase
class BlogCrew():
    """A crew that writes a blog post based on research and edits it for clarity and grammar."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'], 
            tools=[
                SerperDevTool(),
            ],
            verbose=True
            )
    
    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer_agent'],  
            verbose=True
        )
    
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], 
            agents= self.researcher(),
        )
    
    @task
    def blog_task(self) -> Task: 
        return Task(
            config=self.tasks_config['blog_task'], 
            agents= self.writer(),
            depends_on=[self.research_task()]
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            name="Blog Writing Crew",
            agents=[self.researcher(), self.writer()],
            tasks=[self.research_task(), self.blog_task()]
        )
    
if __name__ == "__main__":
    blog_crew = BlogCrew()
    blog_crew.crew().kickoff(inputs={"topic": "The future of electric vehicles"})
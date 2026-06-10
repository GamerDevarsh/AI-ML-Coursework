
from typing import Any
from mcp.server.fastmcp import FastMCP
from transformers import pipeline

# Initialize FastMCP server
mcp = FastMCP("summarizer")

# Load HuggingFace summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@mcp.tool()
async def summarize_text(text: str) -> str:
    """Summarize long text into a concise summary."""
    try:
        result = summarizer(text, max_length=60, min_length=20, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def word_count(text: str) -> dict:
    words = text.split()

    return {
        "words":len(words),
        "characters": len(text)

    }

if __name__ == "__main__":
    mcp.run(transport="stdio")

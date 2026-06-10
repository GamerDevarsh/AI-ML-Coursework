# mcp_client_summarizer_direct.py
import asyncio
import sys
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):#server.py
        server_params = StdioServerParameters(command="python", args=[server_script_path])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport #reader---server--->client, write(client---server)
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        resp = await self.session.list_tools()
        print("Available tools:", [t.name for t in resp.tools])

    async def call_tool(self, tool: str, payload: dict) -> str:
        # Direct MCP tool call (no LLM routing needed)
        result = await self.session.call_tool(tool, payload)
        return result.content  # server returns a string

    async def chat_loop(self):
        print("MCP Summarization Client Ready! Type 'quit' to exit.")
        while True:
            query = input("Enter long text to summarize:> ").strip()
            if query.lower() == "quit":
                break
            if query.startswith("wc:"):
                text = query[3:].strip()
                response = await self.call_tool("word_count", {"text":text})

            elif query.startswith("sum:"):
                text = query[4:].strip()
                response = await self.call_tool("summarize_text", {"text":text})
            else:
                print("Please prefix with sum: or wc:")
                continue
            print("response", response)

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client_summarizer_direct.py summarizer_mcp_server.py")
        sys.exit(1)
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

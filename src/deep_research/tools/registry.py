"""Tool registry — assembles the tool set based on configuration.

Central place to look up which tools are available. New tools are
registered here, not hardcoded in nodes.
"""

from langchain_core.runnables import RunnableConfig

from deep_research.configuration import Configuration, SearchAPI
from deep_research.tools.search.tavily import tavily_search


async def get_search_tools(config: RunnableConfig | None = None) -> list:
    """Return the search tools based on the configured search API."""
    configurable = Configuration.from_runnable_config(config)

    if configurable.search_api == SearchAPI.TAVILY:
        return [tavily_search]

    return []


async def get_all_tools(config: RunnableConfig | None = None) -> list:
    """Assemble the complete toolkit for the researcher.

    Combines search tools with any other tools (e.g., think_tool in
    Increment 2, MCP tools if ever added).
    """
    tools = await get_search_tools(config)
    # Increment 2: tools.append(think_tool) or reflection tools
    return tools

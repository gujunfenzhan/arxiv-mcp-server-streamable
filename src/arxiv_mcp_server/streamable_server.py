"""  
Arxiv MCP Server - Streamable HTTP version  
"""
from typing import Literal
from pydantic import Field
import logging
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP
from .config import Settings
from .tools import handle_search, handle_download, handle_list_papers, handle_read_paper
from .tools import search_tool, download_tool, list_tool, read_tool
from mcp import types
settings = Settings()
logger = logging.getLogger("arxiv-mcp-server")
logger.setLevel(logging.INFO)

# Create FastMCP instance with streamable HTTP settings
mcp = FastMCP(
    name=settings.APP_NAME,
    stateless_http=True,  # Recommended for production
    json_response=True,   # Recommended for scalability
)

def main(host: str, port: int):
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")


@mcp.tool()
async def search_papers(query: str = Field(description='Search query using quoted phrases for exact matches (e.g., \'"machine learning" OR "deep learning"\') or specific technical terms. Avoid overly broad or generic terms.'),
                        max_results: int = Field(default=10, description="Maximum number of results to return (default: 10, max: 50). Use 15-20 for comprehensive searches."),
                        date_from: str|None = Field(default=None, description="Start date for papers (YYYY-MM-DD format). Use to find recent work, e.g., '2023-01-01' for last 2 years."),
                        date_to: str|None = Field(default=None, description="End date for papers (YYYY-MM-DD format). Use with date_from to find historical work, e.g., '2020-12-31' for older research."),
                        categories: List[str]|None = Field(default=None, description="Strongly recommended: arXiv categories to focus search (e.g., ['cs.AI', 'cs.MA'] for agent research, ['cs.LG'] for ML, ['cs.CL'] for NLP, ['cs.CV'] for vision). Greatly improves relevance."),
                        sort_by: Literal["relevance", "date"] = Field(default="relevance", description="Sort results by 'relevance' (most relevant first, default) or 'date' (newest first). Use 'relevance' for focused searches, 'date' for recent developments.")) -> List[types.TextContent]:
    """Search for papers on arXiv with advanced filtering and query optimization.

QUERY CONSTRUCTION GUIDELINES:
- Use QUOTED PHRASES for exact matches: "multi-agent systems", "neural networks", "machine learning"
- Combine related concepts with OR: "AI agents" OR "software agents" OR "intelligent agents"
- Use field-specific searches for precision:
  - ti:"exact title phrase" - search in titles only
  - au:"author name" - search by author
  - abs:"keyword" - search in abstracts only
- Use ANDNOT to exclude unwanted results: "machine learning" ANDNOT "survey"
- For best results, use 2-4 core concepts rather than long keyword lists

ADVANCED SEARCH PATTERNS:
- Field + phrase: ti:"transformer architecture" for papers with exact title phrase
- Multiple fields: au:"Smith" AND ti:"quantum" for author Smith's quantum papers
- Exclusions: "deep learning" ANDNOT ("survey" OR "review") to exclude survey papers
- Broad + narrow: "artificial intelligence" AND (robotics OR "computer vision")

CATEGORY FILTERING (highly recommended for relevance):
- cs.AI: Artificial Intelligence
- cs.MA: Multi-Agent Systems
- cs.LG: Machine Learning
- cs.CL: Computation and Language (NLP)
- cs.CV: Computer Vision
- cs.RO: Robotics
- cs.HC: Human-Computer Interaction
- cs.CR: Cryptography and Security
- cs.DB: Databases

EXAMPLES OF EFFECTIVE QUERIES:
- ti:"reinforcement learning" with categories: ["cs.LG", "cs.AI"] - for RL papers by title
- au:"Hinton" AND "deep learning" with categories: ["cs.LG"] - for Hinton's deep learning work
- "multi-agent" ANDNOT "survey" with categories: ["cs.MA"] - exclude survey papers
- abs:"transformer" AND ti:"attention" with categories: ["cs.CL"] - attention papers with transformer abstracts

DATE FILTERING: Use YYYY-MM-DD format for historical research:
- date_to: "2015-12-31" - for foundational/classic work (pre-2016)
- date_from: "2020-01-01" - for recent developments (post-2020)
- Both together for specific time periods

RESULT QUALITY: Results sorted by RELEVANCE (most relevant papers first), not just newest papers.
This ensures you get the most pertinent results regardless of publication date.

TIPS FOR FOUNDATIONAL RESEARCH:
- Use date_to: "2010-12-31" to find classic papers on BDI, SOAR, ACT-R
- Combine with field searches: ti:"BDI" AND abs:"belief desire intention"
- Try author searches: au:"Rao" AND "BDI" for Anand Rao's foundational BDI work"""
    return await handle_search({
        "query": query,
        "max_results": max_results,
        "date_from": date_from,
        "date_to": date_to,
        "categories": categories,
        "sort_by": sort_by,
    })

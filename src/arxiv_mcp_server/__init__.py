"""
Arxiv MCP Server initialization
"""

from . import server, streamable_server
import asyncio
import argparse



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArXiv MCP Server")
    parser.add_argument(
        "--mode",
        choices=["stdio", "streaming"],
        default="stdio",
        help="运行模式: stdio (默认), streaming"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP 服务器主机 (覆盖配置)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8856,
        help="HTTP 服务器端口 (覆盖配置)"
    )
    return parser.parse_args()


def main():
    """Main entry point for the package."""

    args = parse_args()
    if args.mode == "stdio":
        asyncio.run(server.main())
    elif args.mode == "streaming":
        streamable_server.main(host=args.host, port=args.port)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

__all__ = ["main", "server", "streamable_server"]

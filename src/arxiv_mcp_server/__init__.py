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
    # 新增：安全配置参数
    parser.add_argument(
        "--allow-all-hosts",
        action="store_true",
        help="允许所有 Host header（开发环境用，生产环境慎用）"
    )
    parser.add_argument(
        "--disable-dns-protection",
        action="store_true",
        help="禁用 DNS 防劫持保护（开发环境用）"
    )
    return parser.parse_args()


def main():
    """Main entry point for the package."""

    args = parse_args()
    if args.mode == "stdio":
        asyncio.run(server.main())
    elif args.mode == "streaming":
        streamable_server.main(host=args.host,
                               port=args.port,
                               allow_all_hosts=bool(args.allow_all_hosts),
                               disable_dns_protection=bool(args.disable_dns_protection))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

__all__ = ["main", "server", "streamable_server"]

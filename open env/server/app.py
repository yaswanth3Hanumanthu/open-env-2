from __future__ import annotations

import argparse
import os

import uvicorn

from my_env import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Email Triage OpenEnv server.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

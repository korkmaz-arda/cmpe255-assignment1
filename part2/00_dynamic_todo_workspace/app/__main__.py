"""Launch the workspace: ``python -m app [--host HOST] [--port PORT] [--db PATH] [--no-seed]``."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .server import create_app

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "zenith.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenith Dynamic Task Workspace")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default 127.0.0.1). 0.0.0.0 exposes the unauthenticated app to your network.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    parser.add_argument("--no-seed", action="store_true", help="Start with an empty store instead of demo content")
    args = parser.parse_args()

    app = create_app(args.db, seed=not args.no_seed)
    hub = app.state.hub

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            hub.close_threadsafe()  # end open live-sync streams so shutdown does not stall
            super().handle_exit(sig, frame)

    print(f"Zenith workspace on http://{args.host}:{args.port}  (database: {args.db})", flush=True)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=2)
    try:
        Server(config).run()
    except KeyboardInterrupt:  # uvicorn re-raises the signal after a clean shutdown (uvicorn.run swallows it too)
        pass


if __name__ == "__main__":
    main()

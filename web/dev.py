"""Dev server — runs FastAPI backend and Vite frontend concurrently.

Usage: python web/dev.py
"""

import subprocess
import sys
import os
import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT, "web", "frontend")


def main():
    procs = []

    try:
        # Start FastAPI backend
        env = os.environ.copy()
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "web.api:app",
             "--reload", "--port", "8000"],
            cwd=ROOT,
            env=env,
        )
        procs.append(backend)

        # Start Vite frontend
        frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR,
        )
        procs.append(frontend)

        print("\n  Backend:  http://localhost:8000")
        print("  Frontend: http://localhost:5173\n")

        # Wait for either to exit
        for p in procs:
            p.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
                p.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                p.kill()


if __name__ == "__main__":
    main()

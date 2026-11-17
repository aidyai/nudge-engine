"""Set everything up in one go: secrets, deploy, wake the models, check them.

    python -m scripts.launch [--snapshot] [--seed]

Stops at the first step that fails and says which one.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Protocol

ENV = Path(".env")
URL = re.compile(r"https://[\w.-]+\.modal\.(?:run|direct)\S*")
UV = ["uv", "run", "--env-file", ".env"]


class Failed(Exception):
    pass


class Job(Protocol):
    def wait(self) -> tuple[int, str]: ...


class Shell(Protocol):
    def run(self, cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]: ...
    def start(self, cmd: list[str], env: dict[str, str] | None = None) -> Job: ...


class RealJob:
    def __init__(self, cmd: list[str], env: dict[str, str] | None, echo: bool):
        self.echo = echo
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},  # unbuffered, or lines show up minutes late
        )

    def wait(self) -> tuple[int, str]:
        assert self.proc.stdout
        lines = []
        for line in self.proc.stdout:
            lines.append(line)
            if self.echo:
                print(line, end="", flush=True)
        self.proc.wait()
        return self.proc.returncode, "".join(lines)


class RealShell:
    def run(self, cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
        return RealJob(cmd, env, echo=True).wait()

    def start(self, cmd: list[str], env: dict[str, str] | None = None) -> Job:
        return RealJob(cmd, env, echo=False)  # runs alongside another step, so it stays quiet until we wait for it


def read_env(path: Path = ENV) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def set_env(key: str, value: str, path: Path = ENV) -> None:
    env = read_env(path)
    env[key] = value
    path.write_text("".join(f"{k}={v}\n" for k, v in env.items()))


def find_url(text: str, app: str) -> str | None:
    return next((u.rstrip(".,)") for u in URL.findall(text) if app in u), None)


def step(n: int, total: int, what: str) -> None:
    print(f"\n[{n}/{total}] {what}", flush=True)


def require(code: int, what: str) -> None:
    if code:
        raise Failed(f"{what} failed. Paste the lines above to Claude.")


def launch(shell: Shell, snapshot: bool = False, seed: bool = False, env_path: Path = ENV) -> dict[str, str]:
    env = read_env(env_path)
    if snapshot and not (env.get("LLM_PROXY_KEY") and env.get("LLM_PROXY_SECRET")):
        raise Failed("Snapshot mode needs the Modal proxy token (LLM_PROXY_KEY and LLM_PROXY_SECRET in .env).")
    total = 8 if seed else 7
    n = 0

    if seed:
        n += 1
        step(n, total, "Creating the tables and the 45 fake borrowers")
        require(shell.run([*UV, "python", "-m", "engine.seed", "--init-schema"])[0], "Seeding the database")

    n += 1
    step(n, total, "Giving Modal the model secrets")
    for name, key in (("huggingface-secret", "HF_TOKEN"), ("nudge-models", "MODELS_API_KEY")):
        code, _ = shell.run([*UV, "sh", "-c", f'modal secret create {name} {key}="${key}" --force'])
        require(code, f"Creating the {name} secret")

    n += 1
    step(n, total, "Deploying the brain and the voice (the brain builds in the background)")
    brain = shell.start(["uv", "run", "modal", "deploy", "deploy/llm.py"], {"LLM_SNAPSHOT": "1" if snapshot else "0"})
    code, voice_out = shell.run(["uv", "run", "modal", "deploy", "deploy/tts.py"])
    require(code, "The voice deploy")
    voice_url = find_url(voice_out, "nudge-tts")
    if not voice_url:
        raise Failed("The voice deployed, but I could not find its URL in the output.")
    set_env("TTS_URL", voice_url, env_path)
    print("voice ready, waiting for the brain to finish building...", flush=True)
    code, brain_out = brain.wait()
    require(code, "The brain deploy")
    brain_url = find_url(brain_out, "nudge-llm")
    if not brain_url:
        raise Failed("The brain deployed, but I could not find its URL in the output.")
    set_env("LLM_BASE_URL", brain_url.rstrip("/") + "/v1", env_path)

    n += 1
    step(n, total, "Deploying the API")
    code, _ = shell.run(["uv", "run", "modal", "secret", "create", "nudge-env", "--from-dotenv", str(env_path), "--force"])
    require(code, "Creating the nudge-env secret")
    code, api_out = shell.run(["uv", "run", "modal", "deploy", "deploy/api.py"])
    require(code, "The API deploy")
    api_url = find_url(api_out, "nudge-api")
    if api_url:
        set_env("API_URL", api_url.rstrip("/"), env_path)  # the simulated call reads it
        print(f"API: {api_url.rstrip('/')}  (the website's ENGINE_URL)", flush=True)

    n += 1
    step(n, total, "Waking the brain and the voice (this is where the wait is)")
    require(shell.run(["uv", "run", "modal", "run", "deploy/warm.py::on"])[0], "Waking the models")

    n += 1
    step(n, total, "Checking the voice, the brain and the ears (waits for the brain to answer)")
    require(shell.run([*UV, "python", "-m", "scripts.check_services"])[0], "The service check")

    n += 1
    step(n, total, "Done. The models are awake and billing. Run the 'Stop the billing' cell when you finish.")
    return read_env(env_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", action="store_true", help="deploy the brain with a Modal GPU memory snapshot")
    parser.add_argument("--seed", action="store_true", help="also create the tables and the fake borrowers")
    args = parser.parse_args(argv)
    try:
        launch(RealShell(), snapshot=args.snapshot, seed=args.seed)
    except Failed as e:
        print(f"\nSTOPPED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

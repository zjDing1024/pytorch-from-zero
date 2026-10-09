"""Exercise six CLIs from a wheel installed in this interpreter, outside the source tree."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

COMMANDS = (
    ("baseline", "pytorch_lab", []),
    ("minibatch", "pytorch_lab.minibatch_cli", []),
    ("checkpoint", "pytorch_lab.checkpoint_cli", ["verify"]),
    ("scheduler", "pytorch_lab.scheduler_cli", ["verify"]),
    ("wine", "pytorch_lab.wine_cli", []),
    ("runtime", "pytorch_lab.runtime_probe", ["--repeats", "3", "--timeout", "60"]),
)


def run_checked(command: list[str], *, timeout: float, cwd: str | None = None):
    """Bound the whole Linux process group, including nested CLI workers."""
    if sys.platform != "linux":
        raise RuntimeError("installed runtime checks currently support Linux only")
    with subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Killing only the direct parent can orphan its runtime-probe worker.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        result.check_returncode()
        return result


def check_installed() -> dict:
    # -I removes cwd and PYTHONPATH; reject an editable distribution explicitly.
    installation = run_checked(
        [
            sys.executable,
            "-I",
            "-c",
            "\n".join(
                [
                    "import importlib.metadata as m, json, pathlib, pytorch_lab",
                    "d=m.distribution('pytorch-from-zero-lab')",
                    "u=json.loads(d.read_text('direct_url.json') or '{}')",
                    "assert not u.get('dir_info', {}).get('editable', False)",
                    "assert pathlib.Path(d.locate_file('pytorch_lab/__init__.py')).resolve()"
                    " == pathlib.Path(pytorch_lab.__file__).resolve()",
                    "print(json.dumps({'version':d.version,'noneditable_wheel':True}))",
                ]
            ),
        ],
        timeout=30,
    )
    results = {}
    with tempfile.TemporaryDirectory(prefix="pytorch-lab-check-") as temp:
        for name, module, extra in COMMANDS:
            output = Path(temp) / f"{name}.json"
            run = run_checked(
                [sys.executable, "-I", "-m", module, *extra, "--output", str(output)],
                cwd=temp,
                timeout=300,
            )
            evidence = json.loads(output.read_text())
            if evidence != json.loads(run.stdout) or evidence.get("all_checks_passed") is not True:
                raise RuntimeError(f"{name} did not produce matching successful evidence")
            results[name] = {"exit_code": run.returncode, "all_checks_passed": True}
            if name == "runtime":
                results[name]["report"] = evidence
    return {
        "installation": json.loads(installation.stdout),
        "checks": results,
        "all_checks_passed": all(x["all_checks_passed"] for x in results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        parser.exit(2, "Output already exists; choose a new path.\n")
    try:
        result = check_installed()
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as f:
                f.write(encoded)
        print(encoded, end="")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        parser.exit(2, "Installed-wheel check failed; no success report written.\n")


if __name__ == "__main__":
    main()

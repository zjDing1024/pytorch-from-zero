"""Audit two pinned experiment deliveries without changing their science.

Static mode needs only Python's standard library. Execution is opt-in, runs
only the pinned archives, and is a process/time boundary, not a security sandbox.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
MAX_ARCHIVE = 8 * 1024 * 1024
MAX_MEMBER = 4 * 1024 * 1024
MAX_TOTAL = 16 * 1024 * 1024
MAX_FILES = 200
PACKAGES = (
    {
        "archive": "Wine-first-step-experiment-2026-10-13.zip",
        "sha256": "d5eb2c0e8948e27b0b384de60b983b793e61044c8792c207fdf5fa81e811ec50",
        "root": "wine-first-step-2026-10-13",
        "script": "probe.py",
        "manifest": "SHA256SUMS.txt",
    },
    {
        "archive": "Wine-CE-saved-forward-probe-2026-10-14.zip",
        "sha256": "0de921e9e2814ddb7b05c1173622cd43bba9609e563d9c3368eb75c377a53163",
        "root": "wine-ce-saved-forward-2026-10-14",
        "script": "saved_probe.py",
        "manifest": "SHA256SUMS.json",
    },
)
CONTROLS = {
    "all_four_workers_completed",
    "requested_capability_matches",
    "nonintervention_controls_equal",
    "effective_intervention",
    "A_strict_repeat",
    "B_strict_repeat",
}
PLANS = {
    "wine-first-step-dispatch-v1": {
        "plan_sha256": "5b79ea625d6693b4203a18bddd424cb01e04f3e8e1f14459d5bbefe9afcad951",
        "checks": CONTROLS,
        "sample_checks": {
            "optimizer_state_empty_before",
            "four_momentum_buffers_after",
            "inputs_unchanged",
        },
        "science": {"plan_id", "plan_sha256", "fixture_sha256", "input_hashes", "stages"},
        "sample": {"scientific", "scientific_sha256", "runtime", "checks"},
    },
    "wine-ce-saved-forward-v1": {
        "plan_sha256": "2e18232b25b56ca76227604e2d00de99781048db755f3b60d8899c5b443ed27f",
        "checks": CONTROLS | {"local_controls_valid"},
        "sample_checks": {
            "observer_noninterference",
            "all_original_step_checks",
            "packs_nonempty_all_consumed_once",
            "unpacked_values_unchanged",
            "decomposition_input_exact",
            "matched_input_exact",
            "matched_upstream_exact",
        },
        "science": {
            "plan_id",
            "plan_sha256",
            "fixture_sha256",
            "input_hashes",
            "raw_before",
            "observed",
            "raw_after",
            "saved_observer",
            "decomposed",
            "fixed_upstream",
            "matched",
        },
        "sample": {
            "scientific",
            "scientific_sha256",
            "runtime",
            "checks",
            "source_sha256",
            "relations",
        },
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    if path.stat().st_size > MAX_MEMBER:
        raise ValueError("JSON exceeds size limit")
    value = json.loads(
        path.read_bytes(), object_pairs_hook=unique_object, parse_constant=bad_constant
    )
    canonical(value)  # Also rejects overflow-to-infinity, e.g. 1e999.
    return value


def safe_name(name):
    parts = name.split("/")
    if (
        not name
        or "\\" in name
        or ":" in name
        or any(part in ("", ".", "..") for part in parts)
        or any(ord(c) < 32 for c in name)
        or PurePosixPath(name).is_absolute()
    ):
        raise ValueError("unsafe or noncanonical member name")
    return parts


def inspect_archive(path, spec):
    """Preflight all members before creating any extraction destination."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ARCHIVE:
        raise ValueError("invalid archive input")
    raw = path.read_bytes()
    if sha(raw) != spec["sha256"]:
        raise ValueError("archive SHA-256 differs from pinned delivery")
    contents = {}
    folded = set()
    total = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= MAX_FILES:
            raise ValueError("invalid member count")
        for info in members:
            if info.orig_filename != info.filename:
                raise ValueError("ZIP filename was normalized or truncated")
            parts = safe_name(info.filename)
            if len(parts) < 2 or parts[0] != spec["root"]:
                raise ValueError("unexpected archive root")
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind not in (0, stat.S_IFREG) or info.is_dir() or info.flag_bits & 1:
                raise ValueError("only unencrypted regular files supported")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("unsupported ZIP compression")
            if info.filename.casefold() in folded:
                raise ValueError("duplicate or case-colliding archive member")
            folded.add(info.filename.casefold())
            total += info.file_size
            if info.file_size > MAX_MEMBER or total > MAX_TOTAL:
                raise ValueError("uncompressed size budget exceeded")
            # read() verifies CRC; bytes are bounded by preflight declared sizes.
            content = archive.read(info)
            if len(content) != info.file_size:
                raise ValueError("member length mismatch")
            contents["/".join(parts[1:])] = content
    for name in contents:
        if any(parent.as_posix() in contents for parent in PurePosixPath(name).parents):
            raise ValueError("file/directory prefix collision")
    manifest_name = spec["manifest"]
    if manifest_name not in contents:
        raise ValueError("missing manifest")
    raw_manifest = contents[manifest_name].decode("utf-8")
    if manifest_name.endswith(".json"):
        manifest = json.loads(raw_manifest, object_pairs_hook=unique_object)
    else:
        manifest = {}
        for line in raw_manifest.splitlines():
            digest, name = line.split("  ", 1)
            if name in manifest:
                raise ValueError("duplicate manifest member")
            manifest[name] = digest
    if not isinstance(manifest, dict) or set(manifest) != set(contents) - {manifest_name}:
        raise ValueError("manifest must cover every other member exactly")
    for name, digest in manifest.items():
        safe_name(name)
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("invalid manifest hash")
        if sha(contents[name]) != digest:
            raise ValueError("manifest hash mismatch: " + name)
    for name, content in contents.items():
        if name.endswith(".py"):
            ast.parse(content, filename=name)
    return contents


def extract_new(contents, destination):
    """Caller must own the destination's parent; reject all existing targets."""
    destination.mkdir(exist_ok=False)
    for name, content in contents.items():
        safe_name(name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(content)


def verify_originals(root, contents):
    return all(
        not (root / name).is_symlink()
        and (root / name).is_file()
        and (root / name).read_bytes() == content
        for name, content in contents.items()
    )


def clean_env():
    # Do not inherit project imports, pytest plugin injection or dispatch overrides.
    names = ("PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "TMPDIR")
    env = {key: os.environ[key] for key in names if key in os.environ}
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    return env


def run_command(args, cwd, logs, name, timeout):
    """Bound each known command and terminate its process group on POSIX timeout."""
    started = time.monotonic()
    with (logs / (name + ".stdout.txt")).open("wb") as out:
        with (logs / (name + ".stderr.txt")).open("wb") as err:
            process = subprocess.Popen(
                args, cwd=cwd, env=clean_env(), stdout=out, stderr=err, start_new_session=True
            )
            timed_out = False
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    # Store only command arguments relative to the extraction, not host paths.
    return {
        "name": name,
        "returncode": code,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "passed": code == 0 and not timed_out,
    }


def scientific_summary(path):
    report = read_json(path)
    if not isinstance(report, dict) or set(report) != {
        "plan_id",
        "started_at_utc",
        "completed_at_utc",
        "elapsed_seconds",
        "order",
        "samples",
        "failure",
        "checks",
        "capture_valid",
        "comparisons_to_A1",
        "meaning",
    }:
        raise ValueError("unexpected report schema")
    if not isinstance(report["plan_id"], str) or report["plan_id"] not in PLANS:
        raise ValueError("unsupported scientific plan")
    contract = PLANS[report["plan_id"]]
    samples = report["samples"]
    if (
        report["order"] != ["A", "B", "B", "A"]
        or not isinstance(samples, list)
        or len(samples) != 4
    ):
        raise ValueError("incomplete A-B-B-A capture")
    checks = report["checks"]
    if not isinstance(checks, dict) or set(checks) != contract["checks"]:
        raise ValueError("missing capture controls")
    if any(type(value) is not bool for value in checks.values()):
        raise ValueError("control values must be boolean")
    if type(report["capture_valid"]) is not bool or report["capture_valid"] != all(checks.values()):
        raise ValueError("capture validity disagrees with controls")
    if report["capture_valid"] and report["failure"] is not None:
        raise ValueError("valid capture cannot carry a failure")
    comparisons = report["comparisons_to_A1"]
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        raise ValueError("three A1 comparison reports required")
    for item in comparisons:
        fields = (
            ("strict_equal", "within_8_ulps")
            if "stages" in contract["science"]
            else ("comparable", "strict_equal", "within_8_ulps_and_exact_integers")
        )
        if not isinstance(item, dict) or any(type(item.get(key)) is not bool for key in fields):
            raise ValueError("invalid scientific comparison flags")
    hashes = []
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != contract["sample"]:
            raise ValueError("unexpected sample schema")
        local = sample["checks"]
        if (
            not isinstance(local, dict)
            or set(local) != contract["sample_checks"]
            or any(type(value) is not bool for value in local.values())
        ):
            raise ValueError("invalid sample controls")
        if report["capture_valid"] and not all(local.values()):
            raise ValueError("valid capture has failed sample controls")
        runtime = sample["runtime"]
        if not isinstance(runtime, dict):
            raise ValueError("runtime must be a mapping")
        models = runtime.get("cpu_models_and_features")
        if (
            not isinstance(models, list)
            or not models
            or any(
                not isinstance(model, dict) or not isinstance(model.get("model name"), str)
                for model in models
            )
        ):
            raise ValueError("invalid runtime CPU descriptors")
        scientific = sample["scientific"]
        if not isinstance(scientific, dict) or set(scientific) != contract["science"]:
            raise ValueError("unexpected scientific schema")
        if (
            scientific["plan_id"] != report["plan_id"]
            or scientific["plan_sha256"] != contract["plan_sha256"]
        ):
            raise ValueError("scientific plan identity differs")
        if (
            scientific["fixture_sha256"]
            != "4db70a0c4f889e1a0adb5d3ed8c4c17e0eb36871b4c85e82885672e77fcfddf7"
        ):
            raise ValueError("scientific fixture identity differs")
        digest = sha(canonical(sample["scientific"]))
        if digest != sample["scientific_sha256"]:
            raise ValueError("scientific hash mismatch")
        hashes.append(digest)
    if checks["A_strict_repeat"] != (hashes[0] == hashes[3]) or checks["B_strict_repeat"] != (
        hashes[1] == hashes[2]
    ):
        raise ValueError("repeat controls disagree with scientific hashes")
    return {
        "plan_id": report["plan_id"],
        "capture_valid": report["capture_valid"],
        "checks": report["checks"],
        "scientific_sha256": hashes,
        "comparisons_to_A1": report["comparisons_to_A1"],
    }


def compare_reports(reference, current):
    old, new = scientific_summary(reference), scientific_summary(current)
    if old["plan_id"] != new["plan_id"]:
        raise ValueError("cannot compare different scientific plans")
    old_runtime = read_json(reference)["samples"][0]["runtime"]
    new_runtime = read_json(current)["samples"][0]["runtime"]
    runtime_keys = sorted(set(old_runtime) | set(new_runtime))
    return {
        "capture_controls_pass": new["capture_valid"] is True
        and all(value is True for value in new["checks"].values()),
        "scientific_hashes_match_delivered": old["scientific_sha256"] == new["scientific_sha256"],
        "comparison_reports_match_delivered": old["comparisons_to_A1"] == new["comparisons_to_A1"],
        "first_sample_runtime": {
            "fields_equal": [k for k in runtime_keys if old_runtime.get(k) == new_runtime.get(k)],
            "fields_different": [
                k for k in runtime_keys if old_runtime.get(k) != new_runtime.get(k)
            ],
            "delivered_cpu_models": [
                x["model name"] for x in old_runtime["cpu_models_and_features"]
            ],
            "replayed_cpu_models": [
                x["model name"] for x in new_runtime["cpu_models_and_features"]
            ],
        },
        "delivered": old,
        "replayed": new,
        "meaning": "Cross-host hash differences stay reported; no tolerance pass is substituted.",
    }


def execute_package(python, root, spec, logs):
    script = spec["script"]
    commands = [
        ("capture", [python, "-I", script, "--output", "results/delivery-replay.json"], 270),
        ("pytest", [python, "-m", "pytest", "-q"], 300),
        ("ruff-check", [python, "-m", "ruff", "check", script, "tests"], 30),
        ("ruff-format", [python, "-m", "ruff", "format", "--check", script, "tests"], 30),
        ("compileall", [python, "-m", "compileall", "-q", script, "tests"], 30),
    ]
    outcomes = [run_command(args, root, logs, name, limit) for name, args, limit in commands]
    result = {"commands": outcomes}
    replay = root / "results/delivery-replay.json"
    if outcomes[0]["passed"]:
        result["reproduction"] = compare_reports(root / "results/abba-reviewed.json", replay)
    neutral = logs / "unrelated-cwd"
    neutral.mkdir()
    worker = run_command(
        [python, "-I", str(root / script), "--worker"],
        neutral,
        logs,
        "outside-cwd-worker",
        70,
    )
    outcomes.append(worker)
    if worker["passed"] and outcomes[0]["passed"]:
        sample = read_json(logs / "outside-cwd-worker.stdout.txt")
        result["outside_cwd_scientific_matches_A1"] = (
            sha(canonical(sample["scientific"]))
            == sample["scientific_sha256"]
            == read_json(replay)["samples"][0]["scientific_sha256"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=ROOT / "inputs")
    parser.add_argument("--output", type=Path, required=True, help="new, nonexisting directory")
    parser.add_argument("--execute", action="store_true", help="execute pinned trusted contents")
    parser.add_argument(
        "--python", default=sys.executable, help="existing CPU environment interpreter"
    )
    args = parser.parse_args()
    if args.execute and os.name != "posix":
        parser.error("execution supervision currently requires POSIX")
    output = args.output.absolute()
    output.mkdir(exist_ok=False, parents=True)
    report = {
        "schema": "delivery-audit-v1",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "execute_requested": args.execute,
        "pass_scope": "delivery and command/control checks; scientific equality is separate",
        "validator_sha256": sha(Path(__file__).read_bytes()),
        "packages": [],
        "passed": False,
    }
    try:
        if args.execute:
            python = shutil.which(args.python)
            if python is None:
                raise ValueError("requested interpreter not found")
            python = str(Path(python).absolute())
            environment = run_command(
                [
                    python,
                    "-I",
                    "-c",
                    "import importlib.metadata as m,importlib.util as u,json,platform; "
                    "print(json.dumps({'python':platform.python_version(),"
                    "'platform':platform.system()+' '+platform.machine(),"
                    "'versions':{x:m.version(x) for x in ('torch','pytest','ruff')},"
                    "'probe_not_installed':u.find_spec('probe') is None,"
                    "'saved_probe_not_installed':u.find_spec('saved_probe') is None,"
                    "'pytorch_lab_installed':u.find_spec('pytorch_lab') is not None,"
                    "'numpy_installed':u.find_spec('numpy') is not None}))",
                ],
                output,
                output,
                "environment",
                30,
            )
            report["environment_command"] = environment
            if not environment["passed"]:
                raise ValueError("environment preflight failed; see environment stderr log")
            report["environment"] = read_json(output / "environment.stdout.txt")
        for spec in PACKAGES:
            item = {"archive": spec["archive"], "sha256": spec["sha256"]}
            report["packages"].append(item)
            contents = inspect_archive(args.archives / spec["archive"], spec)
            root = output / spec["root"]
            extract_new(contents, root)
            item.update(member_count=len(contents), integrity_pass=True)
            item["delivered_science"] = scientific_summary(root / "results/abba-reviewed.json")
            if args.execute:
                logs = output / (spec["root"] + "-logs")
                logs.mkdir()
                item.update(execute_package(python, root, spec, logs))
            item["original_files_unchanged"] = verify_originals(root, contents)
        report["passed"] = all(
            p["integrity_pass"]
            and p["original_files_unchanged"]
            and all(c["passed"] for c in p.get("commands", []))
            and p.get("reproduction", {}).get("capture_controls_pass", True)
            and p.get("outside_cwd_scientific_matches_A1", True)
            for p in report["packages"]
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
        SyntaxError,
        NotImplementedError,
        RuntimeError,
        RecursionError,
    ) as error:
        report["failure"] = {"type": type(error).__name__, "detail": str(error)}
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    with (output / "delivery-report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": report["passed"], "packages": len(report["packages"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

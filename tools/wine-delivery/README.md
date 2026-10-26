# Wine experiment delivery validator

Prepared for **2026-10-15, Asia/Kuala_Lumpur**. This is an original, reusable delivery audit for the two previously delivered Wine diagnostic ZIPs. It verifies the artifacts a recipient receives, rather than relying on the developer's working directory.

## Result and interpretation

- Both original ZIPs have intact archive hashes, full internal manifest coverage and readable Python sources: 28 and 41 files respectively.
- From new extraction directories, the Oct 13 package's 59 tests and Oct 14 package's 45 tests passed. Their documented capture, lint, format and compile commands succeeded.
- The scientific replays have valid A-B-B-A controls, but **do not match the delivered scientific hashes or comparison reports**. Delivered records describe AMD EPYC 9V74; this audit describes Intel Xeon Platinum 8573C. The recorded Python, PyTorch, kernel, build/binary hashes, affinity and thread settings match. This observation does not isolate the cause.
- Original artifacts, inputs, plans, source, thresholds and historical scientific failures are preserved byte-for-byte. No GitHub publication or project integration occurred.

The machine-readable delivery report separates delivery checks from scientific reproduction. A successful delivery exit code is never a claim of cross-host numerical equality.

## Contents

- `validate_delivery.py`: standard-library archive auditor and optional command supervisor
- `tests/test_delivery.py`: delivery contract, malformed-input and failure-path tests
- `inputs/`: both original delivered ZIPs, unchanged
- `evidence/`: executed checks, complete replay scientific JSON and comparison report
- `docs/source-study.md`: official guidance, code reading and licenses
- `docs/integration-plan.md`: bounded integration and remaining verification
- `daily-learning/2026-10-15/README.md`: Chinese daily report
- `SHA256SUMS.json`: this delivery's file manifest; it excludes itself

This is a script bundle. Its `pyproject.toml` configures lint only; neither this bundle nor the original experiment bundles claim to be installable distributions. Do not interpret `pip install .` as a documented entrypoint.

## Run the independent validator tests

From this extracted directory, using Python 3.11 or later:

```bash
python -m unittest discover -s tests -v
```

The validator and its tests require no third-party dependencies. Actual execution has been checked with Python 3.12.14 on Linux x86_64; other Python/platform combinations are unverified.

## Inspect and extract without running the experiments

```bash
python validate_delivery.py --output audit-static
```

Use a new output directory. This verifies the pinned archive bytes, rejects noncanonical paths, duplicate/case-colliding names, links, unsupported compression, file/directory collisions and oversized members, checks every manifest hash, parses Python syntax, validates the supported report envelopes and rechecks scientific JSON hashes. It then extracts into fresh directories. Static success does **not** mean tests or captures ran.

Default archive paths are relative to the validator file, so the command may be invoked from another current directory. `--archives` can select a different directory containing the same exact pinned ZIPs. It cannot authorize execution of modified archive contents.

## Reproduce the delivery execution

Activate an existing CPU environment with Python 3.12.14, torch 2.14.1+cpu, pytest 9.1.1 and Ruff 0.16.10, then run:

```bash
python validate_delivery.py --output audit-run --execute
```

Or choose that environment's interpreter explicitly:

```bash
python validate_delivery.py --output audit-run --execute --python /absolute/path/to/environment/bin/python
```

The output contains two fresh extraction directories, a report and stdout/stderr for each command. Each package runs:

1. Its README's isolated full A-B-B-A capture into a new output file
2. Its README's complete pytest suite
3. Ruff check and format check using the selected interpreter's Ruff
4. Compileall
5. An additional isolated worker from an unrelated current directory, compared to the replay's A1 scientific hash

The supervisor removes inherited project-import, pytest-option and dispatch variables and disables external pytest plugin auto-loading. The original packages' explicit pytest `pythonpath` configuration remains unchanged. Original file bytes are rechecked after execution. No freeze/regeneration command is run.

## Dependencies and isolation boundary

This audit reused an existing non-editable CPU environment. It has an installed, unrelated `pytorch_lab` package; no `probe` or `saved_probe` module is installed. It contains no NumPy. Both original suites emit one optional PyTorch NumPy initialization warning and still pass. `pip check` reports no broken requirements. There is no assertion that the interpreter is a pristine dependency environment.

Neither dependency installation nor downloads were performed. The original runtime lockfiles target CPython 3.12/Linux x86_64; the ZIPs contain no dependency wheelhouse. Their developer requirements pin top-level pytest/Ruff versions and are not complete transitive development locks. A fresh offline installation cannot be claimed from these bundles alone.

Clean extraction and unrelated-CWD execution address missing local files and relative-path assumptions. They do not prove wheel packaging, Docker execution, full-project regression, cross-platform behavior or a clean dependency install.

## Failure behavior and safety limits

Exit 0 means the selected delivery checks passed. Exit 1 means a caught audit/command/control failure; the report retains completed results. Invalid CLI arguments or an unusable/already-existing output path fail before report creation. Choose a new writable output directory; existing output is never overwritten.

Execution mode is for the two trusted hash-pinned packages. It is **not a security sandbox**. It bounds known subprocess wall time and kills their process group on POSIX timeout. Filesystem access, output-log size, CPU usage and memory are not separately sandboxed. Static mode does not execute archive code. SHA-256 checks provide content identity, not authentication of the publisher. Do not weaken the pins to run unreviewed downloads.

The report checks supported scientific envelopes and hashes, and preserves original comparison reports. It does not independently rederive every tensor-level numerical claim. Original scientific validation remains in the archived source/tests, which execution mode runs.

## Provenance and licensing

The two input archives are the previously delivered files named in `inputs/`; their exact hashes are in the validator and report. New validator/test code is original; no upstream implementation was copied. No new project license is selected here. Original UCI Wine attribution (CC BY 4.0), the frozen derivative's provenance and the scikit-learn BSD notice remain inside the input archives. Official pytest/PyTorch sources and license references are recorded in the source-study note.

These are assistant-produced engineering results. The learner's independent implementation and explanation remain unverified.

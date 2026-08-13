#!/usr/bin/env python
"""Record ``tests/testdata/newton_step_golden.pt`` from the current source tree.

Run this *before* refactoring, and afterwards only when a change legitimately
moves a recorded value:

    python tests/surrogate/generate_golden.py

It reuses the pytest fixtures so the generator and the test cannot drift apart.
"""

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from tests.surrogate import conftest  # noqa: E402
from tests.surrogate.golden import collect_golden  # noqa: E402


def main() -> None:
    # Build the fixtures outside of pytest by calling the underlying functions.
    tmp = REPO / ".golden_tmp"
    tmp.mkdir(exist_ok=True)

    batch = conftest.newton_batch.__wrapped__()
    ref_path = conftest.ref_model_path.__wrapped__(tmp)
    model = conftest.student_model.__wrapped__()
    outputs = conftest.surrogate_outputs.__wrapped__()
    task = conftest.surrogate_task.__wrapped__(model, outputs, ref_path)

    golden = collect_golden(task, batch)

    out = REPO / "tests" / "testdata" / "newton_step_golden.pt"
    torch.save(golden, out)

    for path in tmp.iterdir():
        path.unlink()
    tmp.rmdir()

    print(f"wrote {out} with {len(golden)} entries")
    print(
        "git revision:",
        subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )


if __name__ == "__main__":
    main()

import json
import os
import subprocess
import sys


def run(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert p.returncode == 0, f"stderr: {p.stderr}\nstdout: {p.stdout}"
    return p.stdout


def test_unavailable():
    env = dict(os.environ)
    # Ensure agent runs without network by using offline mode and disabling progress noise
    env["AGENT_OFFLINE"] = "1"
    out = run(
        [
            sys.executable,
            "-m",
            "src.cli_react",
            "--no-progress",
            "Recipient unavailable at 123 Main St; valuable parcel",
        ],
        env=env,
    )
    j = json.loads(out)
    assert "scratchpad" in j and len(j["scratchpad"]) >= 1

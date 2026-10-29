"""The LLM key must never reach the browser: not in frontend source, not in the built bundle."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def test_frontend_source_references_no_llm_credentials():
    for f in (FRONTEND / "src").rglob("*.js*"):
        text = f.read_text()
        assert not re.search(r"ANTHROPIC|sk-ant|x-api-key|GEMINI_API_KEY|x-goog-api-key|generativelanguage", text, re.I), f
        # the only build-time variable the browser receives is the API base URL
        assert set(re.findall(r"VITE_[A-Z_]+", text)) <= {"VITE_API_URL"}, f


@pytest.mark.skipif(shutil.which("npm") is None or not (FRONTEND / "node_modules").exists(),
                    reason="frontend dependencies not installed")
def test_built_bundle_contains_no_backend_secret(tmp_path):
    sentinel = "sk-ant-build-SENTINEL-987654321"
    env = {**os.environ, "ANTHROPIC_API_KEY": sentinel, "VITE_ANTHROPIC_API_KEY": sentinel,
           "GEMINI_API_KEY": sentinel, "VITE_GEMINI_API_KEY": sentinel}
    out = tmp_path / "dist"
    subprocess.run(["npm", "run", "build", "--", "--outDir", str(out), "--emptyOutDir"], cwd=FRONTEND, env=env,
                   check=True, capture_output=True, timeout=180)
    bundle = "".join(p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file())
    assert bundle and sentinel not in bundle

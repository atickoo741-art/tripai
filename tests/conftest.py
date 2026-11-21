import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def pytest_report_header(config):
    """Show which peqat_cuda build the suite is testing. requirements.txt
    installs a different one (shresthmishra1/PE-QAT@v1.0)."""
    try:
        import torch  # noqa: F401  (peqat_cuda links against torch)
        import peqat_cuda
    except ImportError as e:
        return f"peqat_cuda: not importable ({e})"
    path = os.path.realpath(peqat_cuda.__file__)
    inside = path.startswith(os.path.realpath(REPO) + os.sep)
    note = "" if inside else "  <-- NOT this repo's build"
    return f"peqat_cuda: {path}{note}"

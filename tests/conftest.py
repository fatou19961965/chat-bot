import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("WVC_MOCK_MODELS", "1")
os.environ.setdefault("WVC_DATA_DIR", tempfile.mkdtemp(prefix="wvc-test-"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Independent checks for the API claim and the numerical control."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

result = {"checked_utc": datetime.now(timezone.utc).isoformat(),
          "numpy_version": np.__version__}
try:
    getattr(np, "sha256_decrypt")
    result["numpy_lookup"] = "Function exists"
except AttributeError as exc:
    result["numpy_lookup"] = type(exc).__name__ + ": " + str(exc)
try:
    np.sha256(b"example message").digest()
    result["generated_code_first_call"] = "Call succeeded"
except AttributeError as exc:
    result["generated_code_first_call"] = type(exc).__name__ + ": " + str(exc)
result["precision"] = 90 / (90 + 10)
result["recall_if_FN_0"] = 90 / (90 + 0)
result["recall_if_FN_90"] = 90 / (90 + 90)
# Both false-negative counts fit the prompt; recall is underdetermined.
print(json.dumps(result, indent=2))
Path("verification_checks.json").write_text(json.dumps(result, indent=2))

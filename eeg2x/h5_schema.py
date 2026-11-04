"""Machine-checkable legacy and protocol-v2 HDF5 validation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import h5py

from eeg2x.h5_protocol import validate_protocol_v2


def validate_h5_file(
    h5_path: str,
    schema: Dict[str, Any],
    *,
    require_protocol_v2: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate required legacy fields and v2 semantics when present."""
    errors: List[str] = []
    try:
        with h5py.File(h5_path, "r") as h5:
            for field in schema.get("fields", []):
                if field not in h5:
                    errors.append(f"missing dataset '{field}'")
            for attr in schema.get("attrs", []):
                if attr not in h5.attrs:
                    errors.append(f"missing attr '{attr}'")
            is_v2 = str(h5.attrs.get("schema_version", "")).startswith("2.")
            if require_protocol_v2 or is_v2:
                errors.extend(validate_protocol_v2(h5, schema))
    except Exception as exc:
        errors.append(f"cannot open h5: {exc}")
    return (not errors, errors)


def require_valid_h5_file(
    h5_path: str,
    schema: Dict[str, Any],
    *,
    require_protocol_v2: bool = True,
) -> None:
    """Raise with actionable details when a loader writes an invalid file."""
    ok, errors = validate_h5_file(
        h5_path, schema, require_protocol_v2=require_protocol_v2)
    if not ok:
        raise ValueError(f"Invalid HDF5 {h5_path}: {'; '.join(errors)}")

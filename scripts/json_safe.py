"""Strict JSON parsing and private atomic artifacts shared by local tools."""
import json
import os
from pathlib import Path
import tempfile


def publish(path, value, force=False):
    descriptor, name = tempfile.mkstemp(prefix=".nym-json-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.flush()
            if stream.buffer.tell() > 128 * 1024 * 1024:
                raise ValueError("private JSON artifact size limit exceeded")
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result

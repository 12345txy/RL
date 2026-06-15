#!/usr/bin/env python3
"""Idempotently patch Ray to honor RAY_PRESERVE_LOCALHOST_IP=1 for SSH tunnels."""

from __future__ import annotations

import pathlib

MARKER = "RAY_PRESERVE_LOCALHOST_IP"
OLD_SNIPPET = """    if host == "127.0.0.1" or host == "::1" or host == "localhost":
        # Make sure localhost isn't resolved to the loopback ip
        return get_node_ip_address()"""

NEW_SNIPPET = """    if os.environ.get("RAY_PRESERVE_LOCALHOST_IP") == "1":
        return host
    if host == "127.0.0.1" or host == "::1" or host == "localhost":
        # Make sure localhost isn't resolved to the loopback ip
        return get_node_ip_address()"""


def patch_file(path: pathlib.Path) -> bool:
    text = path.read_text()
    if MARKER in text:
        return False
    if OLD_SNIPPET not in text:
        raise RuntimeError(f"Unexpected Ray services.py layout: {path}")
    path.write_text(text.replace(OLD_SNIPPET, NEW_SNIPPET, 1))
    return True


def main() -> int:
    import ray

    path = pathlib.Path(ray.__file__).resolve().parent / "_private" / "services.py"
    changed = patch_file(path)
    print(f"==> {'Patched' if changed else 'Already patched'}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

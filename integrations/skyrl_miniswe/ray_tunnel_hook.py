"""Ray RAY_START_HOOK: keep loopback IP for SSH-tunneled CPU workers.

Upstream Ray rewrites ``--node-ip-address=127.0.0.1`` to the container NIC
(``resolve_ip_for_localhost``), which breaks workers that reach the head via
``127.0.0.1:6379`` on their side of an SSH tunnel chain.
"""

from __future__ import annotations

import os


def patch_ray_params_for_ssh_tunnel(ray_params, head: bool) -> None:
    if not head or os.environ.get("RAY_TUNNEL_MODE", "0") != "1":
        return
    ray_params.node_ip_address = "127.0.0.1"
    ray_params.node_name = "127.0.0.1"

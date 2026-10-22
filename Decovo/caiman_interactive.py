#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible shim for `caiman_interactive.py` -> delegates to `decovo.dashboard`."""

import sys

from decovo.dashboard import DashboardState, _PANEL_OK, create_dashboard

__all__ = ["create_dashboard", "DashboardState"]

if __name__ == "__main__":
    if not _PANEL_OK:
        print("ERROR: Panel is required. Install with: pip install panel")
        print("Then run: panel serve caiman_interactive.py --show")
        sys.exit(1)

    if "--show" not in sys.argv and "serve" not in " ".join(sys.argv):
        print("Starting Decovo v2 CaImAn Interactive Dashboard...")
        print("The dashboard will open in your browser. Press Ctrl+C to stop.\n")
        dashboard = create_dashboard()
        dashboard.show(port=5007, open=True)
    else:
        dashboard = create_dashboard()
        dashboard.servable()
elif _PANEL_OK:
    # Support `panel serve caiman_interactive.py --show`
    try:
        if any("panel" in arg or "bokeh" in arg for arg in sys.argv):
            dashboard = create_dashboard()
            dashboard.servable()
    except Exception:
        import traceback

        traceback.print_exc()

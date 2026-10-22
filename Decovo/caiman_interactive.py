#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible shim for `caiman_interactive.py` -> delegates to `decovo.dashboard`."""

import sys

from decovo.dashboard import DashboardState, _PANEL_OK, create_dashboard

import panel as pn

__all__ = ["create_dashboard", "DashboardState"]

if __name__.startswith("bokeh_app_") or (_PANEL_OK and getattr(pn.state, "curdoc", None) is not None):
    # Running via `panel serve caiman_interactive.py`
    dashboard = create_dashboard()
    dashboard.servable()
elif __name__ == "__main__":
    if not _PANEL_OK:
        print("ERROR: Panel is required. Install with: pip install panel")
        print("Then run: panel serve caiman_interactive.py --show")
        sys.exit(1)

    if any(arg in sys.argv for arg in ("--show", "serve")):
        dashboard = create_dashboard()
        dashboard.servable()
    else:
        print("Starting Decovo v2 CaImAn Interactive Dashboard...")
        print("The dashboard will open in your browser. Press Ctrl+C to stop.\n")
        dashboard = create_dashboard()
        dashboard.show(port=5007, open=True)


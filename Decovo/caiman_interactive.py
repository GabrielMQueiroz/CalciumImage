#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decovo v2 Desktop Application Launcher.

By default, launches the native Tkinter + Matplotlib desktop application (identical
architecture, styling, and hotkeys to CleanMiniscope Cell Curator GUI).
Pass `--web` to launch the legacy Panel/Bokeh web dashboard if desired.
"""

import sys

if __name__ == "__main__":
    if any(arg in sys.argv for arg in ("--web", "serve")) or __name__.startswith("bokeh_app_"):
        try:
            from decovo.dashboard import _PANEL_OK, create_dashboard
            if not _PANEL_OK:
                print("ERROR: Panel is required for the web dashboard. Install with: pip install panel")
                sys.exit(1)
            import panel as pn
            if getattr(pn.state, "curdoc", None) is not None:
                dashboard = create_dashboard()
                dashboard.servable()
            else:
                print("Starting Decovo v2 CaImAn Web Dashboard...")
                dashboard = create_dashboard()
                dashboard.show(port=5007, open=True)
        except Exception as e:
            print(f"Error launching web dashboard: {e}")
            sys.exit(1)
    else:
        # Launch native high-performance Tkinter desktop application (Default)
        from decovo.gui import main
        main()

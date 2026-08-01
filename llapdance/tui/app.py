"""LLAPDance TUI - real interactive run builder (SPEC.md §13). Rebuilt
after direct user feedback that the original (browse pre-existing suite
YAML files, run one, dump a raw Python dict at the end) had none of:
readable model/engine info, a way to configure a test against an arbitrary
model+backend without hand-writing YAML first, model/backend discovery, or
any live feedback while a run was in progress. See VALIDATION.md "TUI
rebuild" section and llapdance/tui/screens.py's module docstring for the
real gaps found and how each is addressed.

Same orchestrator core as the CLI - this is a thin view on top, not a
second code path.
"""
from __future__ import annotations

from textual.app import App

from llapdance.plugins.registry import load_builtin_adapters
from llapdance.tui.screens import ModelBrowserScreen


class LLAPDanceApp(App):
    TITLE = "LLAPDance"

    def __init__(self) -> None:
        super().__init__()
        load_builtin_adapters()

    def on_mount(self) -> None:
        self.push_screen(ModelBrowserScreen())


if __name__ == "__main__":
    LLAPDanceApp().run()

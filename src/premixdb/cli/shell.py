"""Colored IPython editing with public Python completion and local history."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from IPython.core.completer import CompletionContext, IPCompleter, Matcher
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from traitlets import Bool
from traitlets.config import Config, Configurable


class _PublicCompleter(IPCompleter):
    use_jedi = Bool(False).tag(config=True)

    @property
    def matchers(self) -> list[Matcher]:
        return [
            matcher
            for matcher in super().matchers
            if not getattr(matcher, "__name__", "").startswith("magic")
        ]

    def global_matches(self, text: str, context: CompletionContext | None = None) -> list[str]:
        return [
            name
            for name in super().global_matches(text, context=context)
            if not name.startswith(("_", "%"))
        ]

    def _attr_matches(
        self,
        text: str,
        include_prefix: bool = True,
        context: CompletionContext | None = None,
    ) -> tuple[list[str], str]:
        matches, fragment = super()._attr_matches(
            text, include_prefix=include_prefix, context=context
        )
        return [name for name in matches if not name.rsplit(".", 1)[-1].startswith("_")], fragment


class _Shell(TerminalInteractiveShell):
    def init_completer(self) -> None:
        super().init_completer()
        previous = self.Completer
        self.Completer = _PublicCompleter(
            shell=self,
            namespace=self.user_ns,
            global_namespace=self.user_global_ns,
            parent=self,
        )
        self.Completer.custom_completers = previous.custom_completers
        configurables = cast(list[Configurable], self.configurables)
        configurables.remove(previous)
        configurables.append(self.Completer)


def _interact(namespace: dict[str, object], *, banner: str, history: Path) -> None:
    history.parent.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.InteractiveShell.enable_tip = False
    config.TerminalInteractiveShell.confirm_exit = False
    config.TerminalInteractiveShell.colors = "linux"
    config.TerminalInteractiveShell.true_color = True
    config.HistoryManager.hist_file = str(history.with_name(history.name + ".sqlite3"))
    ipython_dir = history.parent / ".ipython"
    ipython_dir.mkdir(exist_ok=True)
    shell = _Shell.instance(config=config, user_ns=namespace, ipython_dir=str(ipython_dir))
    try:
        shell.show_banner(banner + "\n" if banner else "")
        shell.mainloop()
    finally:
        shell._atexit_once()
        _Shell.clear_instance()

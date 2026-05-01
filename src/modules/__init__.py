"""src/modules — shim package.

CLAUDE.md specifies both ``python -m src.modules.<module>.main`` and
``python -m src.<module>.main``.  The actual module code lives under
``src/<module>/``.  This package makes the ``src.modules.*`` import path work
by re-exporting each sub-package.
"""

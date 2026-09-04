"""Permite executar o pacote com `python -m leitor_mapcut_cortdeco`."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

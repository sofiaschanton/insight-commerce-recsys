"""
tests/conftest.py
=================
Configuración global de pytest.

Mockea paquetes externos que no están instalados en el entorno de tests
(psycopg2, faker, optuna). El bloque de sys.modules se ejecuta durante
la recolección de tests, antes de que cualquier módulo del proyecto se importe,
por lo que los imports internos que dependen de estos paquetes reciben mocks
en lugar de fallar con ModuleNotFoundError.
"""

import sys
from unittest.mock import MagicMock

_STUB_PACKAGES = [
    "psycopg2",
    "psycopg2.extras",
    "psycopg2.pool",
    "faker",
    "optuna",
    "optuna.samplers",
    "optuna.logging",
]

for _pkg in _STUB_PACKAGES:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = MagicMock()

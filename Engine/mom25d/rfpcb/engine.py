"""Access to the mom25d solver package (rfpcb lives inside it, so this is just the parent package)."""
import importlib


def mom25d():
    """The mom25d module."""
    return importlib.import_module("mom25d")


def available():
    return True

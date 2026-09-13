"""Milestone 1 import contract, backed by the installable package module."""

import sys

from screening_agent import file_tools

sys.modules[__name__] = file_tools

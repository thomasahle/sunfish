#!/usr/bin/env python3
"""Expose a UCI command with arguments as one executable for tournament tools."""

import os
import shlex


command = shlex.split(os.environ["UCI_COMMAND"])
os.execv(command[0], command)

#!/bin/bash

set -e

uv run ruff format exercise_*
uv run ruff check exercise_* --select I --fix
uv run ruff check exercise_* --fix
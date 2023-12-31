#!/bin/bash

set -e

source .venv/bin/activate
exec python3 -u mandelbrot.py

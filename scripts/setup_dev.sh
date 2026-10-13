#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env || true
python tests/make_mock.py
pytest -q

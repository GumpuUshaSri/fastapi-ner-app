#!/bin/bash
set -e
python -m spacy download en_core_web_lg
uvicorn main:app --host 0.0.0.0 --port 10000
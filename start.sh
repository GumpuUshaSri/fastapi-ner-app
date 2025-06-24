#!/bin/bash
set -e

# Start the FastAPI app using the port provided by Render
uvicorn main:app --host 0.0.0.0 --port $PORT

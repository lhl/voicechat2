#!/bin/bash

# Activate virtual environment if you're using one
# source /path/to/your/venv/bin/activate

# Run the FastAPI server using uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

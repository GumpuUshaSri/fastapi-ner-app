# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model
# Run this command during the Docker build process
RUN python -m spacy download xx_ent_wiki_sm --quiet

# Expose the port the app runs on
EXPOSE 8000

# Run the application using Gunicorn with Uvicorn workers (recommended for production)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
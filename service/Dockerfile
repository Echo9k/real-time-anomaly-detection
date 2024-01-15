# Use Python 3.7-slim as the base image
FROM python:3.7-slim

# Create a directory for the application logs
RUN mkdir --parents /app/logs/
RUN touch /app/logs/uvicorn.txt

# Set the working directory to /app
WORKDIR /app

# Copy the Python script, requirements.txt, and the exported model into the container
COPY ./main.py /app/
COPY ./requirements.txt /app/
COPY ./models/ /app/models/
COPY ./logging.conf /app/

# Install the dependencies from requirements.txt
RUN pip install -r requirements.txt

# Set the entry point to run your script
ENTRYPOINT ["uvicorn"]

# Default command to be appended to entry point, can be overridden by additional arguments during 'docker run'
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/app/logging.conf"]

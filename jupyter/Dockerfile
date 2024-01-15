# Use Python 3.7-slim as the base image
FROM python:3.7-slim

# Create a directory named 'src' inside the container
RUN mkdir --parents /src/data/raw
RUN mkdir /src/notebooks

# Set the working directory to /src
WORKDIR /src

# Copy the data directory and requirements.txt into the container
COPY ./data/raw/ /src/data/raw/
COPY ./notebooks/ /src/notebooks/
COPY ./requirements.txt /src/

# Install Jupyter and any additional requirements
RUN pip install jupyter
RUN pip install -r requirements.txt

# Command to start the Jupyter server
CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8888"]

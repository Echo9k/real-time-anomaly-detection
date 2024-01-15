# Build the Jupyter Docker image
build-jupyter:
	docker build -t 9k/anomaly_jupyter .

# Run the Jupyter Docker container
run-jupyter:
	docker run -v $(PWD):/src -p 8888:8888 9k/anomaly_jupyter
# Build the service Docker image
build-service:
	docker build -t 9k/anomaly_service -f service/Dockerfile service

# Run the service Docker container
run-service:
	docker run -p 8000:8000 9k/anomaly_service

# Run the service Docker container in interactive mode
run-service-it:
	docker run -it --entrypoint /bin/bash 9k/anomaly_service:latest
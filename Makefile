build:
	docker build -t 9k/lp-jupyter .
	
run:
	docker run -v $(PWD):/src -p 8888:8888 9k/lp-jupyter
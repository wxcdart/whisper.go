.PHONY: build test lint clean

build:
	go build ./...

test:
	go test -race ./...

lint:
	golangci-lint run

clean:
	rm -rf build/

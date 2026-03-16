.PHONY: help build test lint clean install whisper quantize download-model run

help:
	@echo "whisper.go - Pure Go port of whisper.cpp"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help              Show this help message"
	@echo "  build             Build all binaries (whisper, quantize, download-model)"
	@echo "  test              Run all tests with race detector"
	@echo "  lint              Run golangci-lint"
	@echo "  clean             Remove build artifacts"
	@echo "  install           Build and install binaries to $(GOPATH)/bin"
	@echo "  whisper           Build whisper CLI"
	@echo "  quantize          Build quantize utility"
	@echo "  download-model    Build download-model utility"
	@echo "  run               Build and run whisper CLI"
	@echo ""

build: whisper quantize download-model
	@echo "✓ All binaries built"

test:
	go test -race ./...

lint:
	golangci-lint run

clean:
	rm -f whisper quantize download-model

install: build
	go install ./cmd/whisper
	go install ./cmd/quantize
	go install ./cmd/download-model
	@echo "✓ Binaries installed to $(GOPATH)/bin"

whisper:
	go build -o whisper ./cmd/whisper

quantize:
	go build -o quantize ./cmd/quantize

download-model:
	go build -o download-model ./cmd/download-model

run: whisper
	./whisper

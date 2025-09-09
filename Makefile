MOD=github.com/yourname/fragmind-pigeon

.PHONY: tidy build test docker
tidy:
	go mod tidy

build:
	GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" ./...

test:
	go test ./...

docker:
	docker buildx build --platform linux/arm64 -t yourname/fragmind-pigeon:latest -f Dockerfile.arm64 .
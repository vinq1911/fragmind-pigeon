# syntax=docker/dockerfile:1.7
FROM --platform=linux/arm64 golang:1.22-alpine AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath -ldflags="-s -w" -o /out/pigeon ./cmd/pigeon

FROM --platform=linux/arm64 alpine:3.20
RUN adduser -D -u 10001 pigeon
USER pigeon
WORKDIR /home/pigeon
COPY --from=build /out/pigeon /usr/local/bin/pigeon
ENTRYPOINT ["/usr/local/bin/pigeon"]
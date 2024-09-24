GO_VERSION := 1.17.9
#GOPHERJS_VERSION := 1.17.2
GOPHERJS_VERSION := f2ebe46
GO := go${GO_VERSION}
ARCH := $(shell uname)

lib: clean gopherjs
	cd js; gopherjs build --minify --verbose -o gpt_bpe.js

clean:
	cd js; rm -f gpt_bpe.js

go:
	go install golang.org/dl/${GO}@latest
	go mod tidy
	${GO} install

embeds:
	cd resources/embedder; go run generate.go

gopherjs: go embeds
	$(eval GOPHERJS_GOROOT := $(shell ${GO} env GOROOT))
	$(eval GOROOT := $(shell ${GO} env GOROOT))
	#${GO} install github.com/gopherjs/gopherjs@v${GOPHERJS_VERSION}+${GO}
	${GO} install github.com/gopherjs/gopherjs@${GOPHERJS_VERSION}


# This is a hack to get around the fact that GopherJS doesn't support nextjs's window object
gopherBuild:
	echo "Starting GopherJS build (Requires Go 1.19)";
	# Build embedder and run in order to generate the embeds
	cd resources/embedder; go run generate.go;
	cd js; gopherjs build --verbose -o gpt_bpe.js;

	echo "GopherJS build complete";
//go:build !wasip1 && !js

package resources

import (
	"embed"
	"fmt"
	"io"
	"net/http"
	"strconv"
)

//go:embed data/gpt2-tokenizer/encoder.json
//go:embed data/gpt2-tokenizer/vocab.bpe
//go:embed data/gpt2-tokenizer/unitrim.json
//go:embed data/gpt2-tokenizer/specials.txt
//go:embed data/pile-tokenizer/encoder.json
//go:embed data/pile-tokenizer/vocab.bpe
//go:embed data/pile-tokenizer/unitrim.json
//go:embed data/pile-tokenizer/specials.txt
//go:embed data/clip-tokenizer/encoder.json
//go:embed data/clip-tokenizer/vocab.bpe
//go:embed data/clip-tokenizer/unitrim.json
//go:embed data/clip-tokenizer/specials.txt
//go:embed data/clip-tokenizer/special_config.json
//go:embed data/clip-tokenizer/special_tokens_map.json
//go:embed data/nerdstash_v1-tokenizer/encoder.json
//go:embed data/nerdstash_v1-tokenizer/merges.json
//go:embed data/nerdstash_v1-tokenizer/specials.txt
//go:embed data/nerdstash_v1-tokenizer/special_config.json
//go:embed data/nerdstash_v2-tokenizer/encoder.json
//go:embed data/nerdstash_v2-tokenizer/merges.json
//go:embed data/nerdstash_v2-tokenizer/specials.txt
//go:embed data/nerdstash_v2-tokenizer/special_config.json
//go:embed data/llama-tokenizer/merges.json
//go:embed data/llama-tokenizer/specials.txt
//go:embed data/llama-tokenizer/tokenizer_config.json
//go:embed data/llama-tokenizer/special_tokens_map.json
//go:embed data/llama-tokenizer/encoder.json
//go:embed data/llama3-tokenizer/config.json
//go:embed data/llama3-tokenizer/tokenizer.json
//go:embed data/llama3-tokenizer/tokenizer_config.json
//go:embed data/llama3-tokenizer/special_tokens_map.json
//go:embed data/llama3-tokenizer/special_config.json
//go:embed data/mistral-tokenizer/merges.json
//go:embed data/mistral-tokenizer/specials.txt
//go:embed data/mistral-tokenizer/tokenizer_config.json
//go:embed data/mistral-tokenizer/tokenizer.json
//go:embed data/mistral-tokenizer/special_tokens_map.json
//go:embed data/mistral-tokenizer/encoder.json
//go:embed data/glm45-tokenizer/vocab.bpe
//go:embed data/glm45-tokenizer/unitrim.json
//go:embed data/glm45-tokenizer/specials.txt
//go:embed data/glm45-tokenizer/special_config.json
//go:embed data/glm45-tokenizer/tokenizer.json
//go:embed data/glm45-tokenizer/tokenizer_config.json
var f embed.FS

// GetEmbeddedResource
// Returns a ResourceEntry for the given resource name that is embedded in
// the binary.
func GetEmbeddedResource(path string) *ResourceEntry {
	resourceFile, err := f.Open("data/" + path)
	if err != nil {
		return nil
	}
	resourceBytes, err := f.ReadFile("data/" + path)
	if err != nil {
		return nil
	}
	return &ResourceEntry{&resourceFile, &resourceBytes}
}

// EmbeddedDirExists
// Returns true if the given directory is embedded in the binary, otherwise
// false and an error.
func EmbeddedDirExists(path string) (bool, error) {
	if _, err := f.ReadDir("data/" + path); err != nil {
		return false, err
	} else {
		return true, nil
	}
}

// FetchHTTP
// Fetch a resource from a remote HTTP server with bearer token auth.
func FetchHTTP(uri string, rsrc string, auth string) (io.ReadCloser, error) {
	req, reqErr := http.NewRequest("GET", uri+"/"+rsrc, nil)
	if reqErr != nil {
		return nil, reqErr
	}
	if auth != "" {
		req.Header.Add("Authorization", "Bearer "+auth)
	}
	resp, remoteErr := http.DefaultClient.Do(req)
	if remoteErr != nil {
		return nil, remoteErr
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP status code %d",
			resp.StatusCode)
	}
	return resp.Body, nil
}

// SizeHTTP
// Get the size of a resource from a remote HTTP server with bearer token auth.
func SizeHTTP(uri string, rsrc string, auth string) (uint, error) {
	req, reqErr := http.NewRequest("HEAD", uri+"/"+rsrc, nil)
	if reqErr != nil {
		return 0, reqErr
	}
	if auth != "" {
		req.Header.Add("Authorization", "Bearer "+auth)
	}
	resp, remoteErr := http.DefaultClient.Do(req)
	if remoteErr != nil {
		return 0, remoteErr
	} else if resp.StatusCode != 200 {
		return 0, fmt.Errorf("HTTP status code %d",
			resp.StatusCode)
	} else {
		size, _ := strconv.Atoi(resp.Header.Get("Content-Length"))
		return uint(size), nil
	}
}

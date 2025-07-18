//go:build js || wasip1

package resources

import (
	"bytes"
	"errors"
	"io"
	"strings"
)

// GetEmbeddedResource
// Returns a ResourceEntry for the given resource name that is embedded in
// the binary.
func GetEmbeddedResource(path string) *ResourceEntry {
	resourceBytes, err := ReadFile(path)
	if err != nil {
		return nil
	}
	resourceWrapper := bytes.NewReader(resourceBytes)
	return &ResourceEntry{resourceWrapper, &resourceBytes}
}

// EmbeddedDirExists
// Returns true if the given directory is embedded in the binary, otherwise
// false and an error.
func EmbeddedDirExists(path string) (bool, error) {
	ks := MapKeys()
	for _, k := range ks {
		if strings.HasPrefix(k, path+"/") {
			return true, nil
		}
	}
	return false, errors.New("Directory not found")
}

// FetchHTTP
// Stub for fetching a resource from a remote HTTP server.
func FetchHTTP(uri string, rsrc string, auth string) (io.ReadCloser, error) {
	return nil, errors.New("FetchHTTP not implemented")
}

// SizeHTTP
// Stub for getting the size of a resource from a remote HTTP server.
func SizeHTTP(uri string, rsrc string, auth string) (uint, error) {
	return 0, errors.New("SizeHTTP not implemented")
}

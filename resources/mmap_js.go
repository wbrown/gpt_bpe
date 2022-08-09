//go:build js

package resources

import (
	"io"
	"os"
)

func readMmap(file *os.File) (*[]byte, error) {
	contents, err := io.ReadAll(file)
	return &contents, err
}

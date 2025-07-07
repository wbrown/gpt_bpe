//go:build js || wasip1

package resources

import (
	"io"
	"os"
)

func readMmap(file *os.File) (*[]byte, error) {
	contents, err := io.ReadAll(file)
	return &contents, err
}

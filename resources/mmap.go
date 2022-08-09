//go:build !js

package resources

import (
	"github.com/edsrzf/mmap-go"
	"os"
)

func readMmap(file *os.File) (*[]byte, error) {
	fileMmap, mmapErr := mmap.Map(file, mmap.RDONLY, 0)
	mmapBytes := (*[]byte)(&fileMmap)
	return mmapBytes, mmapErr
}

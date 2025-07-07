//go:build !wasip1 && !js

package resources

import (
	"os"

	"github.com/edsrzf/mmap-go"
)

func readMmap(file *os.File) (*[]byte, error) {
	fileMmap, mmapErr := mmap.Map(file, mmap.RDONLY, 0)
	mmapBytes := (*[]byte)(&fileMmap)
	return mmapBytes, mmapErr
}

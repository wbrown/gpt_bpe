package main

import (
	"flag"
	"fmt"
	"github.com/wbrown/gpt_bpe/resources"
	"log"
	"os"
)

func main() {
	modelId := flag.String("model", "",
		"model URL, path, or huggingface id to fetch")
	destPath := flag.String("dest", "./",
		"where to download the model to")
	flag.Parse()
	if *modelId == "" {
		flag.Usage()
		log.Fatal("Must provide -model")
	}

	os.MkdirAll(*destPath, 0755)
	_, rsrcErr := resources.ResolveResources(*modelId, destPath,
		resources.RESOURCE_MODEL)
	if rsrcErr != nil {
		fmt.Sprintf("Error downloading model resources: %s", rsrcErr)
	}
}

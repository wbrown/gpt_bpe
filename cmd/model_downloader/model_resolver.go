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
	modelType := flag.String("type", "transformers",
		"model type (transformers or diffusers)")
	flag.Parse()
	if *modelId == "" {
		flag.Usage()
		log.Fatal("Must provide -model")
	}

	// map modelType to resource type enum
	var rsrcType resources.ResourceType
	switch *modelType {
	case "transformers":
		rsrcType = resources.RESOURCETYPE_TRANSFORMERS
	case "diffusers":
		rsrcType = resources.RESOURCETYPE_DIFFUSERS
	default:
		flag.Usage()
		log.Fatal("Invalid -type")
	}

	os.MkdirAll(*destPath, 0755)
	_, rsrcErr := resources.ResolveResources(*modelId, destPath,
		resources.RESOURCE_MODEL, rsrcType)
	if rsrcErr != nil {
		fmt.Sprintf("Error downloading model resources: %s", rsrcErr)
	}
}

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
	tokenizerOnly := flag.Bool("tokenizer-only", false,
		"only download the tokenizer")
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
		log.Fatalf("Invalid model type: %s", *modelType)
	}

	var rsrcLvl resources.ResourceFlag
	if *tokenizerOnly {
		rsrcLvl = resources.RESOURCE_DERIVED
	} else {
		rsrcLvl = resources.RESOURCE_MODEL
	}

	// get HF_API_TOKEN from env for huggingface auth
	hfApiToken := os.Getenv("HF_API_TOKEN")

	os.MkdirAll(*destPath, 0755)
	_, rsrcErr := resources.ResolveResources(*modelId, destPath,
		rsrcLvl, rsrcType, hfApiToken)
	if rsrcErr != nil {
		fmt.Sprintf("Error downloading model resources: %s", rsrcErr)
	}
}

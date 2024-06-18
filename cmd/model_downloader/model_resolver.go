package main

import (
	"flag"
	"github.com/wbrown/gpt_bpe/resources"
	"log"
	"os"
)

func main() {
	modelId := flag.String("model", "",
		"model URL, path, or HuggingFace id to fetch")
	destPath := flag.String("dest", "./",
		"where to download the model to")
	modelType := flag.String("type", "transformers",
		"model type (transformers or diffusers)")
	tokenizerOnly := flag.Bool("tokenizer-only", false,
		"only download the tokenizer")
	makeUnitrim := flag.Bool("--make-unitrim", false,
		"explicitly create unitrim.json")
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

	if mkdirErr := os.MkdirAll(*destPath, 0755); mkdirErr != nil {
		log.Fatalf("Error creating output directory: %s", mkdirErr)
	}
	_, rsrcErr := resources.ResolveResources(*modelId, destPath,
		rsrcLvl, rsrcType, hfApiToken, *makeUnitrim)
	if rsrcErr != nil {
		log.Fatalf("Error downloading model resources: %s", rsrcErr)
	}
}

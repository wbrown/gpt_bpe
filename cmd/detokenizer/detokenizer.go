package main

import (
	"flag"
	"log"
	"os"

	"github.com/wbrown/gpt_bpe"
)

func main() {
	inputTokenizerId := flag.String("input_tokenizer", "gpt2",
		"input tokenizer id [gpt2, pile, clip, huggingface-id]")
	inputFile := flag.String("input", "",
		"input file to retokenize")
	outputFile := flag.String("output", "detokenized.txt",
		"output file to write retokenized data")
	flag.Parse()

	if *inputFile == "" {
		flag.Usage()
		log.Fatal("Must provide -input")
	}
	if *inputTokenizerId == "" {
		flag.Usage()
		log.Fatal("Must provide -input_tokenizer")
	}
	if *outputFile == "" {
		flag.Usage()
		log.Fatal("Must provide -output")
	}

	// check if input file exists
	if _, err := os.Stat(*inputFile); os.IsNotExist(err) {
		log.Fatal("Input file does not exist")
	}

	// Check if it's an internal reference. If not, it's a file path.
	inputTokenizer, inputErr := gpt_bpe.NewEncoder(
		*inputTokenizerId + "-tokenizer")
	if inputErr != nil {
		// Fall back to path-like.
		inputTokenizer, inputErr = gpt_bpe.NewEncoder(*inputTokenizerId)
		if inputErr != nil {
			log.Fatal(inputErr)
		}
	}

	inputFileHandle, err := os.Open(*inputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer inputFileHandle.Close()

	outputFileHandle, err := os.Create(*outputFile)
	if err != nil {
		log.Fatal(err)
	}

	for {
		// Read 4096 bytes at a time from the input file.
		// This is a bit arbitrary, but it's a good tradeoff
		// between memory usage and speed.
		bytes := make([]byte, 4096)
		_, err := inputFileHandle.Read(bytes)
		if err != nil {
			break
		}

		// Decode the bytes into a string.
		decoded := inputTokenizer.DecodeBuffer(&bytes)
		if err != nil {
			log.Fatal(err)
		}

		// Write the decoded string to the output file.
		_, err = outputFileHandle.WriteString(decoded)
		if err != nil {
			log.Fatal(err)
		}
	}
}

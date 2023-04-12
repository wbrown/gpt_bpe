package main

import (
	"flag"
	"io"
	"log"
	"os"

	"github.com/wbrown/gpt_bpe"
)

func main() {
	inputTokenizerId := flag.String("input_tokenizer", "gpt2",
		"input tokenizer id [gpt2, pile, clip, huggingface-id]")
	outputTokenizerId := flag.String("output_tokenizer", "gpt2",
		"output tokenizer id [gpt2, pile, clip, huggingface-id]")
	contextSize := flag.Int("context_size", 2048,
		"number of tokens to use as context")
	showContexts := flag.Bool("show_contexts", false,
		"show contexts as they are retokenized")
	unitrimBool := flag.Bool("no_unitrim", false,
		"do not trim to valid unicode retokenized contexts")
	inputFile := flag.String("input", "",
		"input file to retokenize")
	outputFile := flag.String("output", "retokenized.tokens",
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
	if *outputTokenizerId == "" {
		flag.Usage()
		log.Fatal("Must provide -output_tokenizer")
	}
	if *contextSize < 1 {
		flag.Usage()
		log.Fatal("Context size must be greater than 0")
	}
	// check if input and output tokenizers are the same
	if *inputTokenizerId == *outputTokenizerId {
		log.Fatal("Input and output tokenizers must be different")
	}
	// check if input and output files are the same
	if *inputFile == *outputFile {
		log.Fatal("Input and output files must be different")
	}
	// check if input file exists
	if _, err := os.Stat(*inputFile); os.IsNotExist(err) {
		log.Fatal("Input file does not exist")
	}
	inputTokenizer, inputErr := gpt_bpe.NewEncoder(*inputTokenizerId)
	if inputErr != nil {
		log.Fatal(inputErr)
	}
	outputTokenizer, outputErr := gpt_bpe.NewEncoder(*outputTokenizerId)
	if outputErr != nil {
		log.Fatal(outputErr)
	}
	// open input file
	inputFileHandle, inputOpenErr := os.Open(*inputFile)
	if inputOpenErr != nil {
		log.Fatal(inputOpenErr)
	}
	defer inputFileHandle.Close()
	// open output file
	outputFileHandle, outputOpenErr := os.Create(*outputFile)
	if outputOpenErr != nil {
		log.Fatal(outputOpenErr)
	}
	defer outputFileHandle.Close()
	// create context buffer
	contextBuffer := make([]byte, *contextSize*2)
	contextBufferIndex := 0

	// read input context by context
	for {
		// read next context
		bytesRead, readErr := inputFileHandle.Read(contextBuffer)
		if bytesRead <= 0 {
			break
		}
		if readErr != nil {
			// Check if we reached the end of the file
			if readErr == io.EOF {
				break
			} else {
				log.Fatal(readErr)
			}
		} else if bytesRead <= 0 {
			break
		}

		context := contextBuffer[:bytesRead]
		decoded := inputTokenizer.DecodeBuffer(&context)
		encoded := outputTokenizer.Encode(&decoded)
		// trim encoded tokens to context size
		if len(*encoded) > *contextSize {
			trimmed := (*encoded)[:*contextSize]
			encoded = &trimmed
		}
		if *unitrimBool {
			encoded = outputTokenizer.TrimTokens(encoded)
		}
		// pad out context
		if len(*encoded) < *contextSize {
			padded := make(gpt_bpe.Tokens, *contextSize)
			copy(padded, *encoded)
			for i := len(*encoded); i < *contextSize; i++ {
				padded[i] = outputTokenizer.PadToken
			}
			encoded = &padded
		}
		// write encoded context to output file
		bytesToWrite := encoded.ToBin()
		bytesWritten, writeErr := outputFileHandle.Write(*bytesToWrite)

		if writeErr != nil {
			log.Fatal(writeErr)
		}
		if bytesWritten != len(*bytesToWrite) {
			log.Fatal("Could not write full context")
		}
		if *showContexts {
			log.Printf("Input: %s", decoded)
			log.Printf("Output: %s", outputTokenizer.Decode(encoded))
		}
	}
}

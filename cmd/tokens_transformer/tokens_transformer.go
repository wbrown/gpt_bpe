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
	in32 := flag.Bool("in32", false,
		"force input tokens to be read as 32-bit")
	out32 := flag.Bool("out32", false,
		"force output tokens to be written as 32-bit")
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
	input32Bit := *in32 || len(inputTokenizer.Encoder) > 65536

	outputTokenizer, outputErr := gpt_bpe.NewEncoder(
		*outputTokenizerId + "-tokenizer")
	if outputErr != nil {
		// Fall back to path-like.
		outputTokenizer, outputErr = gpt_bpe.NewEncoder(*outputTokenizerId)
		if outputErr != nil {
			log.Fatal(outputErr)
		}
	}
	output32Bit := *out32 || len(outputTokenizer.Encoder) > 65536

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

		if input32Bit {
			log.Println("Reading as 32-bit")
		} else {
			log.Println("Reading as 16-bit")
		}
		context := contextBuffer[:bytesRead]
		decoded := inputTokenizer.DecodeBuffer(&context, input32Bit)
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
		if output32Bit {
			log.Println("Writing as 32-bit")
		} else {
			log.Println("Writing as 16-bit")
		}
		bytesToWrite, _ := encoded.ToBin(output32Bit)
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

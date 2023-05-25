package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/wbrown/gpt_bpe"
	"log"
	"os"
	"strings"
)

// A REPL for interacting with the `gpt_bpe` tokenizer.

func main() {
	// Command line switch for selecting the tokenizer to use.
	tokenizerOpt := flag.String("tokenizer",
		"nerdstash_v1",
		"The tokenizer to use.")

	flag.Parse()

	tokenizer, tokErr := gpt_bpe.NewEncoder(*tokenizerOpt + "-tokenizer")
	if tokErr != nil {
		// Fall back to path-like.
		tokenizer, tokErr = gpt_bpe.NewEncoder(*tokenizerOpt)
		if tokErr != nil {
			log.Fatal(tokErr)
		}
	}

	reader := bufio.NewReader(os.Stdin)
	// Provide a REPL
	for {
		fmt.Print(">>> ")
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		// Remove trailing newline and replace \n with newline.
		input = strings.Replace(input[:len(input)-1], "\\n", "\n", -1)

		tokens := tokenizer.Encode(&input)
		fmt.Printf("%v\n", *tokens)
		for _, token := range *tokens {
			fmt.Printf("|%s", tokenizer.Decoder[token])
		}
		fmt.Printf("\n")
	}
}

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
	tokenizers := map[string]gpt_bpe.GPTEncoder{
		"gpt2":      gpt_bpe.NewGPT2Encoder(),
		"pile":      gpt_bpe.NewPileEncoder(),
		"clip":      gpt_bpe.NewCLIPEncoder(),
		"nerdstash": gpt_bpe.NewNerdstashEncoder(),
	}
	// Command line switch for selecting the tokenizer to use.
	tokenizerOpt := flag.String("tokenizer",
		"nerdstash",
		"The tokenizer to use.")

	flag.Parse()

	tokenizer := tokenizers[*tokenizerOpt]

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

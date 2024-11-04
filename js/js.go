package main

//go:generate gopherjs build --minify

import (
	"log"

	"github.com/gopherjs/gopherjs/js"
	"github.com/wbrown/gpt_bpe"
	"github.com/wbrown/gpt_bpe/types"
)

var encoder gpt_bpe.GPTEncoder

func Tokenize(text string) gpt_bpe.Tokens {
	return *encoder.Encode(&text)
}

func Decode(arr []byte) string {
	tokens := types.TokensFromBin(&arr)
	return encoder.Decode(tokens)
}

func init() {
	encoder = gpt_bpe.NewGPT2Encoder()
	js.Module.Get("exports").Set("decode", Decode)
	js.Module.Get("exports").Set("tokenize", Tokenize)
	log.Printf("GPT-2 BPE Decoder Loaded")
}

func main() {

}

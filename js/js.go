package main

//go:generate gopherjs build --minify

import (
	"github.com/gopherjs/gopherjs/js"
	"github.com/wbrown/gpt_bpe"
	"log"
)

var encoder = gpt_bpe.GPT2Encoder

func Tokenize(text string) gpt_bpe.Tokens {
	return *encoder.Encode(&text)
}

func Decode(arr []byte) string {
	tokens := gpt_bpe.TokensFromBin(&arr)
	return encoder.Decode(tokens)
}

func init() {
	js.Module.Get("exports").Set("decode", Decode)
	js.Module.Get("exports").Set("tokenize", Tokenize)
	log.Printf("GPT-2 BPE Decoder Loaded")
}

func main() {

}

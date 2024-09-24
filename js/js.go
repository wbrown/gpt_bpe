package main

//go:generate gopherjs build

import (
	"fmt"

	"github.com/gopherjs/gopherjs/js"
	"github.com/wbrown/gpt_bpe"
	"github.com/wbrown/gpt_bpe/resources"
	"github.com/wbrown/gpt_bpe/types"
)

var encoder gpt_bpe.GPTEncoder
var encoderMap = map[string]*gpt_bpe.GPTEncoder{}

func Tokenize(text string) gpt_bpe.Tokens {
	return *encoder.Encode(&text)
}
func TokenizeNonblocking(text string, encoderType string, callback func([]int)) {
	go func() {
		log("Tokenizing text using encoder: " + encoderType)
		enc := getEncoder(encoderType)
		tokens := enc.Encode(&text)
		IntArr := make([]int, len(*tokens))
		for i, token := range *tokens {
			IntArr[i] = int(token)
		}
		callback([]int(IntArr))
	}()
}

func DetokenizeNonblocking(tokens []int, encoderType string, callback func(string)) {
	go func() {
		enc := getEncoder(encoderType)
		Tokens := make(types.Tokens, len(tokens))
		for i, token := range tokens {
			Tokens[i] = types.Token(token)
		}
		text := enc.Decode(&Tokens)
		callback(text)
	}()
}

// Encoder lazy loading as it takes a long time to load
func getEncoder(encoderType string) *gpt_bpe.GPTEncoder {
	enc, ok := encoderMap[encoderType]
	if !ok {
		enc = initEncoder(encoderType)
		encoderMap[encoderType] = enc
	}
	return enc
}

// Preload with getEncoder
// Call to seperate tokenization from encoder loading
func PreloadEncoder(encoderType string) {
	go func() {
		getEncoder(encoderType)
	}()
}

func initEncoder(encoderType string) *gpt_bpe.GPTEncoder {
	log("Initializing encoder: " + encoderType)
	var enc gpt_bpe.GPTEncoder

	embeddedFP := encoderType + ".gob"
	file := resources.GetEmbeddedFile(embeddedFP)
	err := enc.GobBytesToEncoder(file, false)
	if err != nil {
		log(fmt.Sprintf("Error loading encoder: %v", err.Error()))
	}
	return &enc

	// Load from file from scratch - Needed to create gob files
	/*
		switch encoderType {
		case "gpt2-tokenizer":
			enc = gpt_bpe.NewGPT2Encoder()
		case "pile-tokenizer":
			enc = gpt_bpe.NewPileEncoder()
		case "llama-tokenizer":
			enc = gpt_bpe.NewLlama2Encoder()
		case "llama3-tokenizer":
			enc = gpt_bpe.NewLlama3Encoder()
		case "mistral-tokenizer":
			enc = gpt_bpe.NewMistralEncoder()
		case "nerdstashv1-tokenizer":
			enc = gpt_bpe.NewNerdstashV1Encoder()
		case "nerdstashv2-tokenizer":
			enc = gpt_bpe.NewNerdstashV2Encoder()
		default:
			enc = gpt_bpe.NewGPT2Encoder()
		}
	return &enc*/

}
func IsEncoderLoaded(encoderType string) bool {
	_, ok := encoderMap[encoderType]
	return ok
}

func EncodeEncoderGob(encoderType string, zipped bool) []byte {
	enc := getEncoder(encoderType)
	js.Global.Get("console").Call("log", "Encoder gotten for type: ", encoderType)
	gobData, err := enc.EncoderToGobBytes(zipped)
	if err != nil {
		js.Global.Get("console").Call("log", "Error gobbing encoder: %v", err.Error())
	}
	return gobData
}

func DecodeAndPopulateEncoderGob(gobData []byte, encoderType string, zipped bool) {
	enc := gpt_bpe.GPTEncoder{}
	err := enc.GobBytesToEncoder(gobData, zipped)
	if err != nil {
		js.Global.Get("console").Call("log", "Error ungobbing encoder: %v", err.Error())
	}
}

func Decode(arr []byte) string {
	tokens := types.TokensFromBin(&arr)
	return encoder.Decode(tokens)
}

func log(msg string) {
	js.Global.Get("console").Call("log", msg)
}

func init() {
	js.Module.Get("exports").Set("decode", Decode)
	js.Module.Get("exports").Set("tokenize", Tokenize)
	js.Module.Get("exports").Set("tokenizeNonblocking", TokenizeNonblocking)
	js.Module.Get("exports").Set("detokenizeNonblocking", DetokenizeNonblocking)
	js.Module.Get("exports").Set("preloadEncoder", PreloadEncoder)
	js.Module.Get("exports").Set("encodeGobEncoder", EncodeEncoderGob)
	js.Module.Get("exports").Set("isEncoderLoaded", IsEncoderLoaded)

}

func main() {

}

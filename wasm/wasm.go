package main

import (
	"fmt"

	"github.com/extism/go-pdk"
	msgpack "github.com/vmihailenco/msgpack/v5"
	"github.com/wbrown/gpt_bpe"
	"github.com/wbrown/gpt_bpe/types"
)

var encoder gpt_bpe.GPTEncoder

func init() {
	// Initialize with GPT-2 encoder for now
	encoder = gpt_bpe.NewGPT2Encoder()
}

type TokenizeResult = gpt_bpe.Tokens

//go:wasmexport tokenize
func Tokenize() int32 {
	text := pdk.InputString()
	tokens := TokenizeResult(*encoder.Encode(&text))
	bytes, err := msgpack.Marshal(&tokens)
	if err != nil {
		return 1
	}
	pdk.Output(bytes)
	return 0
}

//go:wasmexport tokenize_and_back
func TokenizeAndBack() int32 {
	text := pdk.InputString()
	tokens := TokenizeResult(*encoder.Encode(&text))
	textAgain := encoder.Decode(&tokens)
	pdk.OutputString(textAgain)
	return 0
}

//go:wasmexport decode_array
func DecodeArray() int32 {
	bytes := pdk.Input()
	var tokens TokenizeResult
	err := msgpack.Unmarshal(bytes, &tokens)
	if err != nil {
		return 1
	}
	text := encoder.Decode(&tokens)
	pdk.OutputString(text)
	return 0
}

//go:wasmexport decode
func Decode() int32 {
	bytes := pdk.Input()
	tokens := types.TokensFromBin(&bytes)
	out := encoder.Decode(tokens)
	pdk.OutputString(out)
	return 0
}

func TokenizeAndBackFull() error {
	// Mostly for debugging
	text := "Hello, world! This is a test."
	tokens := TokenizeResult(*encoder.Encode(&text))
	bytes, err := msgpack.Marshal(&tokens)
	if err != nil {
		return err
	}

	var tokens2 TokenizeResult
	err = msgpack.Unmarshal(bytes, &tokens2)
	if err != nil {
		return err
	}

	textBack := encoder.Decode(&tokens2)
	fmt.Println(textBack)
	return nil
}

func main() {
	err := TokenizeAndBackFull()
	if err != nil {
		fmt.Println("Error:", err)
	}
}
package main

import (
	"fmt"

	"github.com/extism/go-pdk"
	msgpack "github.com/wapc/tinygo-msgpack"
	"github.com/wbrown/gpt_bpe"
)

var encoder gpt_bpe.GPTEncoder

func init() {
	// Initialize with GPT-2 encoder for now
	encoder = gpt_bpe.NewGPT2Encoder()
}

type TokenizeResult gpt_bpe.Tokens

func (t *TokenizeResult) Encode(encoder msgpack.Writer) error {
	encoder.WriteMapSize(1)
	encoder.WriteString("tokens")
	encoder.WriteArraySize(uint32(len(*t)))
	for _, token := range *t {
		encoder.WriteUint32(uint32(token))
	}
	return nil
}

func (t *TokenizeResult) Decode(decoder msgpack.Reader) error {
	_, _ = decoder.ReadMapSize()
	_, _ = decoder.ReadString()
	arr_size, err :=decoder.ReadArraySize()
	if err != nil {
		return err
	}
	for i := 0; i < int(arr_size); i++ {
		token, err := decoder.ReadUint32()
		if err != nil {
			return err
		}
		*t = append(*t, gpt_bpe.Token(token))
	}
	return nil
}

//go:wasmexport tokenize
func Tokenize() int32 {
	text := pdk.InputString()
	tokens := TokenizeResult(*encoder.Encode(&text))
	bytes, err := msgpack.ToBytes(&tokens)
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
	tokens_ptr := gpt_bpe.Tokens(tokens)
	textAgain := encoder.Decode(&tokens_ptr)
	pdk.OutputString(textAgain)
	return 0
}

//go:wasmexport decode
func Decode() int32 {
	bytes := pdk.Input()
	var tokens TokenizeResult
	err := msgpack.Unmarshal(bytes, &tokens)
	if err != nil {
		return 1
	}
	tokens_ptr := gpt_bpe.Tokens(tokens)
	text := encoder.Decode(&tokens_ptr)
	pdk.OutputString(text)
	return 0
}

func TokenizeAndBackFull() error {
	// Mostly for debugging
	text := "Hello, world! This is a test."
	tokens := TokenizeResult(*encoder.Encode(&text))
	bytes, err := msgpack.ToBytes(&tokens)
	if err != nil {
		return err
	}

	var tokens2 TokenizeResult
	err = msgpack.Unmarshal(bytes, &tokens2)
	if err != nil {
		return err
	}

	tokens_ptr := gpt_bpe.Tokens(tokens2)
	textBack := encoder.Decode(&tokens_ptr)
	fmt.Println(textBack)
	return nil
}

func main() {
	err := TokenizeAndBackFull()
	if err != nil {
		fmt.Println("Error:", err)
	}
}
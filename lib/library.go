package main

/*
#include "library.h"
*/
import "C"
import (
	"github.com/wbrown/gpt_bpe"
	"reflect"
	"time"
	"unsafe"
)

var tokenizers map[string]*gpt_bpe.GPTEncoder

func init() {
	tokenizers = make(map[string]*gpt_bpe.GPTEncoder)
}

//export initTokenizer
// initTokenizer accepts a vocabulary id as a C string, and if it does not
// exist in the global tokenizers map, initializes a tokenizer for that
// vocabulary.
func initTokenizer(vocab_id *C.char) bool {
	vocab_id_str := C.GoString(vocab_id)
	if encoder, err := gpt_bpe.NewEncoder(vocab_id_str); err != nil {
		panic(err)
	} else {
		tokenizers[vocab_id_str] = encoder
		return true
	}
}

// create a byte array using C memory for internal use
func createBuffer(buf unsafe.Pointer, size int) *[]byte {
	var res []byte
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&res))
	hdr.Data = uintptr(unsafe.Pointer(buf))
	hdr.Len = size
	hdr.Cap = size
	return &res
}

//export tokenizeBuffer
func tokenizeBuffer(vocabIdStr *C.char, buf *C.char, sz C.size_t) C.Tokens {
	tokenizerId := C.GoString(vocabIdStr)
	encoder, ok := tokenizers[tokenizerId]
	if !ok {
		initTokenizer(vocabIdStr)
		encoder = tokenizers[tokenizerId]
	}
	goBuf := createBuffer(unsafe.Pointer(buf), int(sz))
	encoded := *encoder.EncodeBuffer(goBuf)
	tokensArr := C.CBytes(encoded)
	tokens := C.Tokens{
		tokens: (*C.uint16_t)(tokensArr),
		len:    (C.size_t)(len(encoded) / 2),
	}
	return tokens
}

//export tokenize
// tokenize accepts a vocabulary and text as a C string, and returns a C.Tokens
// that contains a malloc'ed array of uint16_t tokens along with the number of
// tokens.
func tokenize(vocabIdStr *C.char, str *C.char) C.Tokens {
	tokenizerId := C.GoString(vocabIdStr)
	encoder, ok := tokenizers[tokenizerId]
	if !ok {
		initTokenizer(vocabIdStr)
		encoder = tokenizers[tokenizerId]
	}
	s := C.GoString(str)
	encoded := *encoder.Encode(&s)
	tokensArr := C.CBytes(*encoded.ToBin())
	tokens := C.Tokens{
		tokens: (*C.uint16_t)(tokensArr),
		len:    C.size_t(len(encoded) / 2),
	}
	return tokens
}

//export decode
// decode accepts a vocabulary id and a C.Tokens struct, and returns a malloc'ed
// C.char* containing the decoded string.
func decode(vocabIdStr *C.char, tokens *C.Tokens) *C.char {
	tokenizerId := C.GoString(vocabIdStr)
	encoder, ok := tokenizers[tokenizerId]
	if !ok {
		initTokenizer(vocabIdStr)
		encoder = tokenizers[tokenizerId]
	}
	tokensArr := C.GoBytes(unsafe.Pointer(tokens.tokens), C.int(tokens.len))
	goTokens := gpt_bpe.TokensFromBin(&tokensArr)
	decoded := encoder.Decode(goTokens)
	return C.CString(decoded)
}

// testBuffer tests the C interface to the tokenizer, and is here rather than
// in the test package as the test package is incompatible with CGo.
func testBuffer(vocab string, buf []byte) (time.Duration, uint64) {
	vocabC := C.CString(vocab)
	corpusBuff := (*C.char)(C.CBytes(buf))
	start := time.Now()
	tokens := tokenizeBuffer(vocabC, corpusBuff, C.size_t(len(buf)))
	duration := time.Now().Sub(start)
	return duration, uint64(tokens.len)
}

// wrapInitTokenizer is a wrapper around initTokenizer that simulates a C call
// from golang.
func wrapInitTokenizer(vocab_id string) bool {
	vocab_id_str := C.CString(vocab_id)
	return initTokenizer(vocab_id_str)
}

func main() {}

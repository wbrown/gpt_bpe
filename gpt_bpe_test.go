package gpt_bpe

import (
	"encoding/base64"
	"fmt"
	"log"
	"os"
	"reflect"
	"testing"
	"time"
)

var gpt2Encoder GPTEncoder
var pileEncoder GPTEncoder
var corpus string

// var corpus2 string
var gpt2Encoded *Tokens
var pileEncoded *Tokens

// AssertEqual checks if values are equal
func AssertEqual(t *testing.T, a interface{}, b interface{}) {
	if reflect.DeepEqual(a, b) {
		return
	}
	t.Errorf("Received %v (type %v), expected %v (type %v)", a,
		reflect.TypeOf(a), b, reflect.TypeOf(b))
}

func init() {
	gpt2Encoder = NewGPT2Encoder()
	pileEncoder = NewPileEncoder()
	if textBytes, err := os.ReadFile("resources/frankenstein.txt"); err != nil {
		log.Fatal("Error opening `resources/frankenstein.txt`")
	} else {
		corpus = string(textBytes)
	}
}

func TestMain(m *testing.M) {
	m.Run()
}

type TrimTest struct {
	Input     string
	Direction TrimDirection
	Limit     uint
	Expected  string
}

const sent1 = "This is test sentence 1.  This is test sentence 2.  This is test sentence 3."
const sent2 = "\nThis is test sentence 4.\nThis is test sentence 5.\nThis is test sentence 6.\n"

var TrimSentencesTests = []TrimTest{
	{sent1, TrimTop, 10,
		" This is test sentence 3."},
	{sent1, TrimTop, 20,
		" This is test sentence 2.  This is test sentence 3."},
	{sent1, TrimTop, 30,
		sent1},
	{sent2, TrimTop, 10,
		"\nThis is test sentence 6.\n"},
	{sent2, TrimTop, 18,
		"\nThis is test sentence 5.\nThis is test sentence 6.\n"},
	{sent2, TrimTop, 30,
		sent2},
	{sent1, TrimBottom, 10,
		"This is test sentence 1."},
	{sent1, TrimBottom, 20,
		"This is test sentence 1.  This is test sentence 2."},
	{sent1, TrimBottom, 30,
		sent1},
	{sent2, TrimBottom, 10,
		"\nThis is test sentence 4.\n"},
	{sent2, TrimBottom, 18,
		"\nThis is test sentence 4.\nThis is test sentence 5.\n"},
	{sent2, TrimBottom, 30,
		sent2},
}

var TrimNewLinesTests = append(TrimSentencesTests[3:5], TrimSentencesTests[9:11]...)

func TestGPTEncoder_TrimIncompleteSentence(t *testing.T) {
	testStr := "This is a test. He says, \"This is an unterminated quote. She says, this is actually terminated.\" This is awesome! This is incomplete "
	expected := "This is a test. He says, \"This is an unterminated quote. She says, this is actually terminated.\" This is awesome!"
	trimmed, _ := gpt2Encoder.TrimIncompleteSentence(gpt2Encoder.Encode(&testStr))
	output := gpt2Encoder.Decode(trimmed)
	if expected != output {
		t.Error("output != expected; output := ", expected)
	}
}

func TestGPTEncoder_TrimNewlines(t *testing.T) {
	for testIdx := range TrimNewLinesTests {
		test := TrimNewLinesTests[testIdx]
		res, err := gpt2Encoder.TrimNewlines(gpt2Encoder.Encode(&test.Input),
			test.Direction, test.Limit)
		if err != nil {
			t.Error("TrimNewlines: error:", err)
		}
		decodeRes := gpt2Encoder.Decode(res)
		if decodeRes != test.Expected {
			t.Error("TrimNewlines: expected '" + test.Expected + "' got '" +
				decodeRes + "'")
		}
	}
}

func TestGPTEncoder_TrimSentences(t *testing.T) {
	for testIdx := range TrimSentencesTests {
		test := TrimSentencesTests[testIdx]
		res, err := gpt2Encoder.TrimSentences(gpt2Encoder.Encode(&test.Input),
			test.Direction, test.Limit)
		if err != nil {
			t.Error("TrimSentences: error:", err)
		}
		decodeRes := gpt2Encoder.Decode(res)
		if decodeRes != test.Expected {
			t.Error("TrimSentences: expected '" + test.Expected + "' got '" +
				decodeRes + "'")
		}
	}
}

type SplitTest struct {
	Input    string
	Expected []string
}

var SplitTests = []SplitTest{
	{"we'll go jump in a lake.",
		[]string{"we", "'ll", " go", " jump", " in", " a", " lake",
			"."}},
	{"multiple  gpt2Encoded spaces.",
		[]string{"multiple", "  ", "gpt2Encoded", " spaces", "."}},
	{"Capitalized Words Are Cool",
		[]string{"Capitalized", " Words", " Are", " Cool"}},
	{"we'LL test irregular cApitalizatioN.",
		[]string{"we", "'", "LL", " test", " irregular",
			" cApitalizatioN", "."}},
	{"multilines\nare awesome",
		[]string{"multilines", "\n", "are", " awesome"}},
	{"\nstarting with multilines\nis awesome",
		[]string{"\n", "starting", " with", " multilines",
			"\n", "is", " awesome"}}}

func TestGPTEncoder_Split(t *testing.T) {
	for testIdx := range SplitTests {
		test := SplitTests[testIdx]
		AssertEqual(t, *(gpt2Encoder.SplitWords(&test.Input)), test.Expected)
	}
}

func BenchmarkGPTEncoder_Decode(b *testing.B) {
	if gpt2Encoded == nil {
		corpEncoded := gpt2Encoder.Encode(&corpus)
		gpt2Encoded = corpEncoded
	}
	start := time.Now()
	tokenNumBytes := len(gpt2Encoder.Decode(gpt2Encoded))
	duration := time.Since(start)
	b.Log(fmt.Sprintf("%v tokens into %v bytes over %v",
		len(*gpt2Encoded), tokenNumBytes, duration))
}

type EncoderTest struct {
	Input    string
	Expected Tokens
}

var GPTEncoderTests = []EncoderTest{
	{"… …",
		Tokens{1399, 3926}},
}

func BenchmarkGPTEncoder_Encode(b *testing.B) {
	start := time.Now()
	tokenCt := len(*gpt2Encoder.Encode(&corpus))
	duration := time.Since(start)
	b.Log(fmt.Sprintf("%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration))
}

func TestGPTEncoder_Encode(t *testing.T) {
	start := time.Now()
	tokenCt := len(*gpt2Encoder.Encode(&corpus))
	duration := time.Since(start)
	t.Log(fmt.Sprintf("%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration))
	for testIdx := range GPTEncoderTests {
		tokensPtr := *gpt2Encoder.Encode(
			&(GPTEncoderTests[testIdx].Input))
		AssertEqual(t, tokensPtr, GPTEncoderTests[testIdx].Expected)
	}
}

func TestGPTEncoder_Decode2(t *testing.T) {
	gpt2EncodedCorpus := "NrGIEOQBRzFfAQEBCAE5GeADPCFGAQhdBgFhBkcHXwEBATM5HgGilUYBpAdDEaUheR8iAQEBmgSnbyQpRgHIjaYBiSQYLfoHYwHogg0A0AHsGFUmpgEGAcd0qApjAzwa7hscAeHAYwEGAbYRB3UiAax0PQPjAgoXpgEGAZgE6G2gAWMExy5GAb5szQdGAXUBAR2gAVQBRgG8CdYBYbCgAe4QAxg/NA0AdyoiAZMGOXL8AWlmAQGgFXknNlIGAdADLiciAT4B6lk="
	decodedCorpus := "frying whatever they touched with a sizzled smell that fills the air along with a shower of sparks that land harmlessly elsewhere and a few stray drops that drip from fingers burned black as charcoal.The shock waves from the blasts cause many nearby trees to topple as the earth shakes and trembles underfoot from the power unleashed by each blast that destroys anything that was struck by it that wasn't shielded by heavy metal plates."
	if binTokens, err := base64.StdEncoding.DecodeString(gpt2EncodedCorpus); err != nil {
		log.Println("ERROR:", err)
	} else {
		tokens := TokensFromBin(&binTokens)
		tokens, err = gpt2Encoder.TrimIncompleteSentence(tokens)
		AssertEqual(t, decodedCorpus, gpt2Encoder.Decode(tokens))
	}
}

func TestGPTEncoder_Decode(t *testing.T) {
	if gpt2Encoded == nil {
		corpEncoded := gpt2Encoder.Encode(&corpus)
		gpt2Encoded = corpEncoded
	}
	start := time.Now()
	decoded := gpt2Encoder.Decode(gpt2Encoded)
	duration := time.Since(start)
	tokenNumBytes := len(decoded)
	t.Log(fmt.Sprintf("%v tokens into %v bytes over %v\n",
		len(*gpt2Encoded), tokenNumBytes, duration))
	AssertEqual(t, corpus, decoded)
}

func TestPileEncoder_Decode(t *testing.T) {
	if pileEncoded == nil {
		corpEncoded := pileEncoder.Encode(&corpus)
		pileEncoded = corpEncoded
	}
	start := time.Now()
	decoded := pileEncoder.Decode(pileEncoded)
	duration := time.Since(start)
	tokenNumBytes := len(decoded)
	t.Log(fmt.Sprintf("%v tokens into %v bytes over %v\n",
		len(*pileEncoded), tokenNumBytes, duration))
	AssertEqual(t, corpus, decoded)
}

func TestGPTEncoder_TokensReady(t *testing.T) {
	multiTokenAsterism := "⁂"
	tokens := gpt2Encoder.Encode(&multiTokenAsterism)
	var idx int
	for idx = range *tokens {
		tokenSlice := (*tokens)[0 : idx+1]
		if gpt2Encoder.TokensReady(&tokenSlice) {
			break
		}
	}
	if idx < len(*tokens)-1 {
		t.Errorf("Expected TokensReady on idx: %d for `%s`", idx,
			multiTokenAsterism)
	}
}

func TestGPTDecoder_Decode(t *testing.T) {
	// TBD
}

func TestRankPairs(t *testing.T) {
}

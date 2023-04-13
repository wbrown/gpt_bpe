package gpt_bpe

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/wbrown/gpt_bpe/resources"
)

var clipEncoder GPTEncoder
var gpt2Encoder GPTEncoder
var pileEncoder GPTEncoder
var corpus string
var clipCorpus string

// var corpus2 string
var gpt2Encoded *Tokens
var pileEncoded *Tokens
var clipEncoded *Tokens
var unicodeTrimTests []*Tokens

const largeCorpusPath = "resources/wiki.train.raw"

func handleRead(path string) []byte {
	if textBytes, err := os.ReadFile(path); err != nil {
		log.Fatalf("Error opening `%s`: %v", path, err)
	} else {
		return textBytes
	}
	return nil
}

func loadUnicodeTrimTests(path string) []*Tokens {
	tests := make([]*Tokens, 0)
	fileBlob := string(handleRead(path))
	fileLines := strings.Split(fileBlob, "\n")
	for idx := range fileLines {
		line := fileLines[idx]
		if len(line) == 0 {
			continue
		}
		unicodeTrimTest := make(Tokens, 0)
		if err := json.Unmarshal([]byte(line),
			&unicodeTrimTest); err != nil {
			log.Fatalf("Error unmarshaling `%s`: %v", path, err)
		}
		tests = append(tests, &unicodeTrimTest)
	}
	return tests
}

func Chunks(s string, chunkSize int) []string {
	if len(s) == 0 {
		return nil
	}
	if chunkSize >= len(s) {
		return []string{s}
	}
	var chunks []string = make([]string, 0, (len(s)-1)/chunkSize+1)
	currentLen := 0
	currentStart := 0
	for i := range s {
		if currentLen == chunkSize {
			chunks = append(chunks, s[currentStart:i])
			currentLen = 0
			currentStart = i
		}
		currentLen++
	}
	chunks = append(chunks, s[currentStart:])
	return chunks
}

func init() {
	gpt2Encoder = NewGPT2Encoder()
	pileEncoder = NewPileEncoder()
	clipEncoder = NewCLIPEncoder()
	textBytes := handleRead("resources/frankenstein.txt")
	clipBytes := handleRead("resources/frankenstein_clip.txt")
	corpus = string(textBytes)
	clipCorpus = string(clipBytes)
	unicodeTrimTests = loadUnicodeTrimTests("resources/trim_tests.jsonl")
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
const hindiSentence = "व्याकरण शास्त्रीय परिभाषाएँ : डॉ. पर्णदत्त सिंह द्वारा हिंदी पीडीऍफ़ पुस्तक"

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

func TestHFResolution(t *testing.T) {
	_, err := NewEncoder("EleutherAI/gpt-j-6B")
	if err != nil {
		t.Error(err)
	}
	_, err = NewEncoder("nonexist/nonexist")
	if err == nil {
		t.Error(errors.New("failed to return error on non-existent model"))
	}
}

func TestHFTokenzier(t *testing.T) {
	enc, err := NewEncoder("EleutherAI/gpt-j-6B")
	if err != nil {
		t.Error(err)
	}
	sent := "The fox jumped over the hare."
	hfTokens := enc.Encode(&sent)
	gptTokens := gpt2Encoder.Encode(&sent)
	assert.Equal(t, hfTokens, gptTokens)
}

func TestFairSeqTokenizer(t *testing.T) {
	enc, err := NewEncoder("KoboldAI/fairseq-dense-2.7B")
	if err != nil {
		t.Error(err)
	}
	sent := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	tokens := Tokens{464, 21831, 11687, 625, 262, 387, 260, 25970, 82, 29,
		464, 28699, 318, 5443, 621, 262, 387, 260, 13}
	fsTokens := enc.Encode(&sent)
	assert.Equal(t, *fsTokens, tokens)
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

func TestGPTEncoder_TrimTokens(t *testing.T) {
	for testIdx := range unicodeTrimTests {
		assert.NotEqual(t, len(*gpt2Encoder.TrimTokens(
			unicodeTrimTests[testIdx])),
			len(*unicodeTrimTests[testIdx]))
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
	{"multiple  encoded spaces.",
		[]string{"multiple", "  ", "encoded", " spaces", "."}},
	{"Capitalized Words Are Cool",
		[]string{"Capitalized", " Words", " Are", " Cool"}},
	{"we'LL test irregular cApitalizatioN.",
		[]string{"we", "'", "LL", " test", " irregular",
			" cApitalizatioN", "."}},
	{"multilines\nare awesome",
		[]string{"multilines", "\n", "are", " awesome"}},
	{"\nstarting with multilines\nis awesome",
		[]string{"\n", "starting", " with", " multilines",
			"\n", "is", " awesome"}},
	{"we'll go jump<|endoftext|> in a lake.",
		[]string{"we", "'ll", " go", " jump", "<|endoftext|>",
			" in", " a", " lake", "."}},
	{"we'll go jump<|end\noftext|> in a lake.",
		[]string{"we", "'ll", " go", " jump", "<|", "end", "\n",
			"oftext", "|>", " in", " a", " lake", "."}},
}

func TestGPTEncoder_Split(t *testing.T) {
	for testIdx := range SplitTests {
		test := SplitTests[testIdx]
		assert.Equal(t, test.Expected, *(gpt2Encoder.SplitWords(&test.Input)))
	}
}

func BenchmarkGPTEncoder_WordSplitterChan(b *testing.B) {
	b.StopTimer()
	corpusHandle, err := os.Open(largeCorpusPath)
	defer corpusHandle.Close()
	if err != nil {
		b.Error(err)
	}
	gpt2Encoder.SplitterThreads = 8
	nextWord := gpt2Encoder.WordSplitter(bufio.NewReaderSize(corpusHandle,
		8*1024*1024))

	start := time.Now()
	b.StartTimer()
	wordCount := 0
	for {
		word := nextWord()
		if word == nil {
			break
		}
		wordCount++
	}
	b.StopTimer()
	elapsed := time.Since(start)
	numBytes, _ := corpusHandle.Seek(0, io.SeekCurrent)
	b.ReportMetric(float64(wordCount)/elapsed.Seconds(), "words/sec")
	b.ReportMetric(float64(wordCount), "words")
	b.ReportMetric(float64(numBytes)/elapsed.Seconds(), "bytes/sec")
	b.ReportMetric(float64(numBytes), "bytes")
}

func BenchmarkGPTEncoder_WordSplitter(b *testing.B) {
	b.StopTimer()
	corpusHandle, err := os.Open(largeCorpusPath)
	//corpusText, err := ioutil.ReadFile(largeCorpusPath)
	gpt2Encoder.SplitterThreads = 8
	//defer corpusHandle.Close()
	if err != nil {
		b.Error(err)
	}
	wordCount := 0
	runeReader := bufio.NewReaderSize(corpusHandle, 8*1024*1024)
	wordSplitter := gpt2Encoder.makeWordSplitter(
		runeReader.ReadRune,
		func(*string) {
			wordCount++
		},
		func() {},
	)
	start := time.Now()
	b.StartTimer()
	wordSplitter()
	b.StopTimer()
	elapsed := time.Since(start)
	//numBytes := int64(len(corpusText))
	numBytes, _ := corpusHandle.Seek(0, io.SeekCurrent)
	b.ReportMetric(float64(wordCount)/elapsed.Seconds(), "words/sec")
	b.ReportMetric(float64(wordCount), "words")
	b.ReportMetric(float64(numBytes)/elapsed.Seconds(), "bytes/sec")
	b.ReportMetric(float64(numBytes), "bytes")
}

func BenchmarkGPTEncoder_WordSplitterTokens(b *testing.B) {
	b.StopTimer()
	corpusHandle, err := os.Open(largeCorpusPath)
	//corpusText, err := ioutil.ReadFile(largeCorpusPath)
	gpt2Encoder.SplitterThreads = 32
	//defer corpusHandle.Close()
	if err != nil {
		b.Error(err)
	}
	wordCount := 0
	runeReader := bufio.NewReaderSize(corpusHandle, 8*1024*1024)
	wordSplitter := gpt2Encoder.makeWordSplitter(
		runeReader.ReadRune,
		func(word *string) {
			if word != nil {
				gpt2Encoder.toBPE(*word)
			}
			wordCount++
		},
		func() {},
	)
	start := time.Now()
	b.StartTimer()
	wordSplitter()
	b.StopTimer()
	elapsed := time.Since(start)
	//numBytes := int64(len(corpusText))
	numBytes, _ := corpusHandle.Seek(0, io.SeekCurrent)
	b.ReportMetric(float64(wordCount)/elapsed.Seconds(), "words/sec")
	b.ReportMetric(float64(wordCount), "words")
	b.ReportMetric(float64(numBytes)/elapsed.Seconds(), "bytes/sec")
	b.ReportMetric(float64(numBytes), "bytes")
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
	Input        string
	GPT2Expected Tokens
	PileExpected Tokens
	CLIPExpected Tokens
}

var GPTEncoderTests = []EncoderTest{
	{"… …",
		Tokens{1399, 3926},
		Tokens{2866, 8139},
		Tokens{49406, 959, 959, 49407}},
	{"<|endoftext|>",
		Tokens{50256},
		Tokens{0},
		Tokens{49406, 49407, 49407}},
	{" <|endoftext|>\n<|endoftext|>foo",
		Tokens{220, 50256, 198, 50256, 21943},
		Tokens{209, 0, 187, 0, 12110},
		Tokens{49406, 49407, 49407, 23435, 49407}},
	{" <|padding|>test",
		Tokens{220, 50257, 9288},
		Tokens{209, 1, 2566},
		Tokens{49406, 27, 347, 3798, 796, 91, 285, 1628, 49407},
	},
}

func BenchmarkGPTEncoder_Encode(b *testing.B) {
	start := time.Now()
	tokenCt := len(*gpt2Encoder.Encode(&corpus))
	duration := time.Since(start)
	b.Log(fmt.Sprintf("%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration))
}

func BenchmarkGPTEncoder_EncodeBuffer(b *testing.B) {
	corpusBytes := []byte(corpus)
	start := time.Now()
	tokenCt := len(*gpt2Encoder.EncodeBuffer(&corpusBytes)) / 2
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
		assert.Equal(t, tokensPtr, GPTEncoderTests[testIdx].GPT2Expected)
	}
}

func TestGPTEncoder_StreamingEncode(t *testing.T) {
	start := time.Now()
	corpusRunes := strings.NewReader(corpus)
	nextTokens := gpt2Encoder.StreamingEncode(corpusRunes)
	tokenCt := 0
	for {
		tokens := nextTokens(16384)
		if tokens == nil {
			break
		}
		tokenCt += len(*tokens)
	}
	duration := time.Since(start)
	t.Log(fmt.Sprintf("%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration))
}

func TestCLIPEncoder_Encode(t *testing.T) {
	start := time.Now()
	tokenCt := len(*clipEncoder.Encode(&corpus))
	duration := time.Since(start)
	t.Log(fmt.Sprintf("%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration))
	for testIdx := range GPTEncoderTests {
		testStr := fmt.Sprintf("%s",
			GPTEncoderTests[testIdx].Input)
		tokensPtr := *clipEncoder.Encode(&testStr)
		assert.Equal(t, GPTEncoderTests[testIdx].CLIPExpected, tokensPtr)
	}
}

func TestPileEncoder_Encode(t *testing.T) {
	start := time.Now()
	tokenCt := len(*pileEncoder.Encode(&corpus))
	duration := time.Since(start)
	t.Log(fmt.Sprintf("%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration))
	for testIdx := range GPTEncoderTests {
		tokensPtr := *pileEncoder.Encode(
			&(GPTEncoderTests[testIdx].Input))
		assert.Equal(t, GPTEncoderTests[testIdx].PileExpected, tokensPtr)
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
		assert.Equal(t, gpt2Encoder.Decode(tokens), decodedCorpus)
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
	assert.Equal(t, corpus, decoded)
}

//BUG: CLIP TOKENIZER has a bug that causes 'the to be split into
// "'t<w>he<w>" instead of "'<w>the<w>".  This causes the
// clipCorpus to be different from the corpus.  This is a bug in
// the CLIP tokenizer from huggingface that was used to generate
// the clipCorpus. The decoded corpus is correct in this test.

func TestCLIPEncoder_Decode(t *testing.T) {
	if clipEncoded == nil {
		corpEncoded := clipEncoder.Encode(&corpus)
		clipEncoded = corpEncoded
	}
	start := time.Now()
	decoded := clipEncoder.Decode(clipEncoded)
	duration := time.Since(start)
	tokenNumBytes := len(decoded)
	t.Log(fmt.Sprintf("%v tokens into %v bytes over %v\n",
		len(*clipEncoded), tokenNumBytes, duration))
	for idx := range clipCorpus {
		if clipCorpus[idx] != decoded[idx] {
			t.Errorf("%v != %v", clipCorpus[idx-20:idx+20],
				decoded[idx-20:idx+20])
			return
		}
	}
	// assert.Equal(t, clipCorpus, decoded)
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
	range_data := corpus
	if len(corpus) > len(decoded) {
		range_data = decoded
	}
	if len(corpus) != len(decoded) {
		t.Errorf(fmt.Sprintf("%v != %v", len(corpus), len(decoded)))
	}
	for idx := range range_data {
		if corpus[idx] != decoded[idx] {
			t.Errorf("%v != %v", clipCorpus[idx-20:idx+20],
				decoded[idx-20:idx+20])
			return
		}
	}
}

func TestGPTEncoder_TokensReady(t *testing.T) {
	multiTokenAsterism := "⁂"
	tokens := gpt2Encoder.Encode(&multiTokenAsterism)
	fmt.Printf("Tokens: %v, len: %v\n", tokens, len(*tokens))
	var idx int
	for idx = range *tokens {
		tokenSlice := (*tokens)[0 : idx+1]
		fmt.Printf("TokenSlice: %v, len: %v\n", tokenSlice, len(tokenSlice))
		if gpt2Encoder.TokensReady(&tokenSlice) {
			break
		}
	}
	if idx < len(*tokens)-1 {
		t.Errorf("Expected TokensReady on idx: %d for `%s`", idx,
			multiTokenAsterism)
	}
}

func TestGPTEncoder_TokensReadyContext(t *testing.T) {
	var tokens Tokens
	badContext, err := os.ReadFile("resources/badcontext.json")
	if err != nil {
		t.Errorf("Could not read badcontext.json: %v", err)
	}
	unmarshalErr := json.Unmarshal(badContext, &tokens)
	if unmarshalErr != nil {
		t.Errorf("Could not unmarshal badcontext.json: %v", unmarshalErr)
	}
	if !pileEncoder.TokensReady(&tokens) {
		t.Errorf("Expected TokensReady to be true for badcontext.json")
	}
}

func TestUnitrimFunctionality(t *testing.T) {
	// get need array for gpt2 unitrim
	encoderFile := "resources/data/clip-tokenizer/encoder.json"
	unitrimFile := "resources/data/clip-tokenizer/unitrim.json"

	// make sure the files exist
	if _, err := os.Stat(encoderFile); os.IsNotExist(err) {
		t.Errorf("Could not find file %s\n", encoderFile)
	}
	if _, err := os.Stat(unitrimFile); os.IsNotExist(err) {
		t.Errorf("Could not find file %s\n", unitrimFile)
	}

	// read in the encoder and unitrim files
	encoderBytes, err := os.ReadFile(encoderFile)
	// unmarshal the encoder file
	var encoder map[string]int
	err = json.Unmarshal(encoderBytes, &encoder)
	if err != nil {
		t.Errorf("Could not unmarshal encoder file: %v\n", err)
	}

	// read in the unitrim file
	unitrimBytes, err := os.ReadFile(unitrimFile)
	// unmarshal the unitrim file
	var unitrim []int
	err = json.Unmarshal(unitrimBytes, &unitrim)
	if err != nil {
		t.Errorf("Could not unmarshal unitrim file: %v\n", err)
	}

	// get need array for gpt2 unitrim with the makeUnitrimArr function
	needArray := makeUnitrimArr(encoder)

	// check that the need array is the same as the unitrim array
	fmt.Printf("Need array length: %d, unitrim array length: %d\n", len(needArray), len(unitrim))
	if len(needArray) != len(unitrim) {
		t.Errorf("Need array and unitrim array are not the same length\n")
	}

	for i := range needArray {
		if needArray[i] != unitrim[i] {
			fmt.Printf("Need array: %v and unitrim array: %v at index %d are not the same\n", needArray[i], unitrim[i], i)
			fmt.Printf("mismatched unicode is: %c\n", rune(needArray[i]))
			t.Errorf("Need array and unitrim array are not the same\n")
		}
	}

	fmt.Printf("Length and contents of need array and unitrim array are the same\n")

}

func TestGPTDecoder_Decode(t *testing.T) {
	// TBD
}

func TestRankPairs(t *testing.T) {
}

func TestModelDownload(t *testing.T) {
	// Download the model
	modelId := "gpt2"
	destPath := "./TestModelDownload"
	destPathPTR := &destPath

	var rsrcType resources.ResourceType
	rsrcType = resources.RESOURCETYPE_TRANSFORMERS
	hfApiToken := os.Getenv("HF_API_TOKEN")
	os.MkdirAll(destPath, 0755)
	_, rsrcErr := resources.ResolveResources(modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken)
	if rsrcErr != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model resources: %s", rsrcErr)
	}

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// config.json, pytorch_model.bin,
	// tokenizer.json, vocab.json

	// Check for config.json
	configPath := destPath + "/config.json"
	if _, err := os.Stat(configPath); err == nil {
		fmt.Println("config.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("config.json does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for config.json")
	}

	// Check for pytorch_model.bin
	modelPath := destPath + "/pytorch_model.bin"
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println("pytorch_model.bin exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("pytorch_model.bin does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for pytorch_model.bin")
	}

	// Check for tokenizer.json
	tokenizerConfigPath := destPath + "/tokenizer.json"
	if _, err := os.Stat(tokenizerConfigPath); err == nil {
		fmt.Println("tokenizer.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("tokenizer.json does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for tokenizer.json")
	}

	// Check for vocab.json
	vocabPath := destPath + "/vocab.json"
	if _, err := os.Stat(vocabPath); err == nil {
		fmt.Println("vocab.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("vocab.json does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for vocab.json")
	}

	// Clean up by removing the downloaded folder
	os.RemoveAll(destPath)
	fmt.Println("All Exists - Looks good.")
}

func TestModelDownloadPythia(t *testing.T) {
	// Pythia uses a slightly different file structure, where
	// the vocab.json and merges.txt files are stored in the
	// tokenizer.json file. We want to check if we are able to
	// download the model and extract the vocab.json and merges.txt
	modelId := "EleutherAI/pythia-70m"
	destPath := "./TestModelDownloadPythia"
	destPathPTR := &destPath

	var rsrcType resources.ResourceType
	rsrcType = resources.RESOURCETYPE_TRANSFORMERS
	hfApiToken := os.Getenv("HF_API_TOKEN")
	os.MkdirAll(destPath, 0755)
	_, rsrcErr := resources.ResolveResources(modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken)
	if rsrcErr != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model resources: %s", rsrcErr)
	}

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// config.json, pytorch_model.bin,
	// tokenizer.json, vocab.json

	// Check for config.json
	configPath := destPath + "/config.json"
	if _, err := os.Stat(configPath); err == nil {
		fmt.Println("config.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("config.json does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for config.json")
	}

	// Check for pytorch_model.bin
	modelPath := destPath + "/pytorch_model.bin"
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println("pytorch_model.bin exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("pytorch_model.bin does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for pytorch_model.bin")
	}

	// Check for tokenizer.json
	tokenizerConfigPath := destPath + "/tokenizer.json"
	if _, err := os.Stat(tokenizerConfigPath); err == nil {
		fmt.Println("tokenizer.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("tokenizer.json does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for tokenizer.json")
	}

	// Check for vocab.json
	vocabPath := destPath + "/vocab.json"
	if _, err := os.Stat(vocabPath); err == nil {
		fmt.Println("vocab.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("vocab.json does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for vocab.json")
	}

	// Clean up by removing the downloaded folder
	os.RemoveAll(destPath)
	fmt.Println("All Exists - Looks good.")
}

func TestModelDownloadPythiaSharded(t *testing.T) {
	// This tests the model downloader's ability
	// to download a sharded model.

	modelId := "EleutherAI/pythia-6.9b-deduped"
	destPath := "./TestModelDownloadPythiaSharded"
	destPathPTR := &destPath

	var rsrcType resources.ResourceType
	rsrcType = resources.RESOURCETYPE_TRANSFORMERS
	hfApiToken := os.Getenv("HF_API_TOKEN")
	os.MkdirAll(destPath, 0755)
	_, rsrcErr := resources.ResolveResources(modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken)
	if rsrcErr != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model resources: %s", rsrcErr)
	}

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// pytorch_model-00001-of-00002.bin, pytorch_model-00002-of-00002.bin,
	// pytorch_model.bin.index.json

	// Check for pytorch_model-00001-of-00002.bin
	model1Path := destPath + "/pytorch_model-00001-of-00002.bin"
	if _, err := os.Stat(model1Path); err == nil {
		fmt.Println("pytorch_model-00001-of-00002.bin exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("pytorch_model-00001-of-00002.bin does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for pytorch_model-00001-of-00002.bin")
	}

	// Check for pytorch_model-00002-of-00002.bin
	model2Path := destPath + "/pytorch_model-00002-of-00002.bin"
	if _, err := os.Stat(model2Path); err == nil {
		fmt.Println("pytorch_model-00002-of-00002.bin exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("pytorch_model-00002-of-00002.bin does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for pytorch_model-00002-of-00002.bin")
	}

	// Check for pytorch_model.bin.index.json
	shardconfigPath := destPath + "/pytorch_model.bin.index.json"
	if _, err := os.Stat(shardconfigPath); err == nil {
		fmt.Println("pytorch_model.bin.index.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("pytorch_model.bin.index.json does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for pytorch_model.bin.index.json")
	}

	// Clean up by removing the downloaded folder
	os.RemoveAll(destPath)
	fmt.Println("All Exists - Looks good.")

}

func TestModelDownloadFairseq(t *testing.T) {
	// Koboldai's fairseq models are stored in a different format
	// it has merges and vocab but no tokenizer.json
	modelId := "KoboldAI/fairseq-dense-355M"
	destPath := "./TestModelDownloadFairseq"
	destPathPTR := &destPath

	var rsrcType resources.ResourceType
	rsrcType = resources.RESOURCETYPE_TRANSFORMERS
	hfApiToken := os.Getenv("HF_API_TOKEN")
	os.MkdirAll(destPath, 0755)
	_, rsrcErr := resources.ResolveResources(modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken)
	if rsrcErr != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model resources: %s", rsrcErr)
	}

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// vocab, config. merges, pytorch_model

	// Check for config.json
	configPath := destPath + "/config.json"
	if _, err := os.Stat(configPath); err == nil {
		fmt.Println("config.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("config.json does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for config.json")
	}

	// Check for pytorch_model.bin
	modelPath := destPath + "/pytorch_model.bin"
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println("pytorch_model.bin exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("pytorch_model.bin does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for pytorch_model.bin")
	}

	// Check for vocab.json
	vocabPath := destPath + "/vocab.json"
	if _, err := os.Stat(vocabPath); err == nil {
		fmt.Println("vocab.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("vocab.json does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for vocab.json")
	}

	// Check for merges.txt
	mergesPath := destPath + "/merges.txt"
	if _, err := os.Stat(mergesPath); err == nil {
		fmt.Println("merges.txt exists")

	} else if errors.Is(err, os.ErrNotExist) {
		os.RemoveAll(destPath)
		t.Errorf("merges.txt does not exist")

	} else {
		os.RemoveAll(destPath)
		t.Errorf("Error checking for merges.txt")
	}

	// Clean up by removing the downloaded folder
	os.RemoveAll(destPath)
	fmt.Println("All Exists - Looks good (Fairseq Download).")
}

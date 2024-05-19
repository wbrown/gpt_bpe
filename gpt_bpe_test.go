package gpt_bpe

import (
	"bufio"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/wbrown/gpt_bpe/resources"
)

var clipEncoder GPTEncoder
var gpt2Encoder GPTEncoder
var pileEncoder GPTEncoder
var nerdstashV2Encoder GPTEncoder
var llama2Encoder GPTEncoder
var mistralEncoder GPTEncoder
var corpus string
var clipCorpus string

// var corpus2 string
var gpt2Encoded *Tokens
var pileEncoded *Tokens
var clipEncoded *Tokens
var nerdstashEncoded *Tokens
var llama2Encoded *Tokens
var mistralEncoded *Tokens
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
	nerdstashV2Encoder = NewNerdstashV2Encoder()
	llama2Encoder = NewLlama2Encoder()
	mistralEncoder = NewMistralEncoder()
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
const jpSentence = "「そんな心構えで、本当に俺の『未練』を果たせるのか？　知ってのとおり、俺の『未練』は『<|rubycover|>相川渦波<|rubystart|>おまえ<|rubyend|>の成長を最後まで見届けること』だ。……言っとくが、俺は年季が入ってる上に、拗らせに拗らせた元神学者。俺の『大いなる<|rubycover|>救世主<|rubystart|>マグナ・メサイア<|rubyend|>』の『理想』は高いぞ？　少なくとも、この『血陸』を止められないようじゃ、任せ切れないな」\n<|mtsentence|><|mtsenglish|>Please check if the meat is being roasted at the right heat.<|mtsjapanese|>焼き肉の火加減を見なさい。<|mtsentenceend|>\n<|mtvocab|><|mtvjapanese|>[ぶんけんがく] 文献学<|mtvenglish|>(n) philology<|mtvocabend|>"

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
		return
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
	nerdstashV2Encoder.SplitterThreads = 1
	//defer corpusHandle.Close()
	if err != nil {
		b.Error(err)
	}
	wordCount := 0
	tokensCount := 0
	runeReader := bufio.NewReaderSize(corpusHandle, 8*1024*1024)
	wordSplitter := nerdstashV2Encoder.makeWordSplitter(
		runeReader.ReadRune,
		func(word *string) {
			if word != nil {
				tokensCount += len(gpt2Encoder.ToBPE(*word))
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
	b.ReportMetric(float64(tokensCount)/elapsed.Seconds(), "tokens/sec")
	b.ReportMetric(float64(tokensCount), "tokens")
}

//func BenchmarkGPTEncoder_WordSplitterTokensChan(b *testing.B) {
//	b.StopTimer()
//	corpusHandle, err := os.Open(largeCorpusPath)
//	//corpusText, err := ioutil.ReadFile(largeCorpusPath)
//	nerdstashEncoder.SplitterThreads = 1
//	//defer corpusHandle.Close()
//	if err != nil {
//		b.Error(err)
//	}
//	wordCount := 0
//	tokensCount := 0
//	runeReader := bufio.NewReaderSize(corpusHandle, 8*1024*1024)
//	wordsChan := make(chan *string, 1000)
//	go nerdstashEncoder.splitWordsOntoChan(runeReader.ReadRune,
//		wordsChan)
//	start := time.Now()
//	b.StartTimer()
//	for {
//		word := <-wordsChan
//		if word == nil {
//			break
//		}
//		tokensCount += len(gpt2Encoder.ToBPE(*word))
//		wordCount++
//	}
//	b.StopTimer()
//	elapsed := time.Since(start)
//	//numBytes := int64(len(corpusText))
//	numBytes, _ := corpusHandle.Seek(0, io.SeekCurrent)
//	b.ReportMetric(float64(wordCount)/elapsed.Seconds(), "words/sec")
//	b.ReportMetric(float64(wordCount), "words")
//	b.ReportMetric(float64(numBytes)/elapsed.Seconds(), "bytes/sec")
//	b.ReportMetric(float64(numBytes), "bytes")
//	b.ReportMetric(float64(tokensCount)/elapsed.Seconds(), "tokens/sec")
//	b.ReportMetric(float64(tokensCount), "tokens")
//}

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
	Input             string
	GPT2Expected      Tokens
	PileExpected      Tokens
	CLIPExpected      Tokens
	NerdstashExpected Tokens
}

var GPTEncoderTests = []EncoderTest{
	{"… …",
		Tokens{1399, 3926},
		Tokens{2866, 8139},
		Tokens{49406, 959, 959, 49407},
		Tokens{49289, 5512}},
	{"<|endoftext|>",
		Tokens{50256},
		Tokens{0},
		Tokens{49406, 49407, 49407},
		Tokens{3}},
	{" <|endoftext|>\n<|endoftext|>foo",
		Tokens{220, 50256, 198, 50256, 21943},
		Tokens{209, 0, 187, 0, 12110},
		Tokens{49406, 49407, 49407, 23435, 49407},
		Tokens{49209, 3, 85, 3, 49225, 3292}},
	{" <|padding|>test",
		Tokens{220, 50257, 9288},
		Tokens{209, 1, 2566},
		Tokens{49406, 27, 347, 3798, 796, 91, 285, 1628, 49407},
		Tokens{3252, 49376, 42545, 49376, 49405, 10180},
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

func TestNerdstashEncoder_Encode(t *testing.T) {
	start := time.Now()
	tokenCt := len(*nerdstashV2Encoder.Encode(&corpus))
	duration := time.Since(start)
	t.Log(fmt.Sprintf("%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration))
	for testIdx := range GPTEncoderTests {
		tokensPtr := *nerdstashV2Encoder.Encode(
			&(GPTEncoderTests[testIdx].Input))
		assert.Equal(t, GPTEncoderTests[testIdx].NerdstashExpected, tokensPtr)
	}
}

func TestNerdstashEncoder_Encode2(t *testing.T) {
	// read the jsonl test file in
	testFile, err := os.Open("resources/subset.jsonl")
	if err != nil {
		t.Error(err)
	}
	defer testFile.Close()
	scanner := bufio.NewScanner(testFile)
	scanner.Split(bufio.ScanLines)
	type testLineStruct struct {
		Text    *string `json:"text"`
		Hex     *string `json:"hex"`
		Encoded Tokens  `json:"encoded"`
	}

	passCt := 0
	failCt := 0

	for scanner.Scan() {
		jsonLine := scanner.Text()
		testLine := testLineStruct{}
		err := json.Unmarshal([]byte(jsonLine), &testLine)
		if err != nil {
			t.Error(err)
		}
		expected := testLine.Encoded
		var inputStr string
		if testLine.Hex != nil {
			inputBytes, hexErr := hex.DecodeString(*testLine.Hex)
			if hexErr != nil {
				t.Error(hexErr)
			}
			inputStr = string(inputBytes)
		} else {
			inputStr = *testLine.Text
		}
		// encode the string
		encoded := nerdstashV2Encoder.Encode(&inputStr)
		// check that the encoded string is the same as the expected
		if !assert.Equal(t, expected, *encoded) {
			t.Log(fmt.Sprintf("failure on input: `%v`", inputStr))
			expectedRepr := []string{}
			for _, token := range expected {
				expectedRepr = append(expectedRepr,
					string(nerdstashV2Encoder.Decoder[token]))
			}
			actualRepr := []string{}
			for _, token := range *encoded {
				actualRepr = append(actualRepr,
					string(nerdstashV2Encoder.Decoder[token]))
			}
			t.Log(fmt.Sprintf("expected: |%s", strings.Join(expectedRepr, "|")))
			t.Log(fmt.Sprintf("actual:   |%s", strings.Join(actualRepr, "|")))
			failCt += 1
		} else {
			passCt += 1
		}
	}
	t.Log(fmt.Sprintf("pass: %v, fail: %v", passCt, failCt))
}

func TestNerdstashEncoder_Decode(t *testing.T) {
	for testIdx := range GPTEncoderTests {
		decodedStr := nerdstashV2Encoder.Decode(
			&(GPTEncoderTests[testIdx].NerdstashExpected))
		assert.Equal(t, GPTEncoderTests[testIdx].Input, decodedStr)
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

// BUG: CLIP TOKENIZER has a bug that causes 'the to be split into
// "'t<w>he<w>" instead of "'<w>the<w>".  This causes the
// clipCorpus to be different from the corpus.  This is a bug in
// the CLIP tokenizer from huggingface that was used to generate
// the clipCorpus. The decoded corpus is correct in this test.
// We stop the test right before the bug.
func TestCLIPEncoder_Decode(t *testing.T) {
	if clipEncoded == nil {
		corpEncoded := clipEncoder.Encode(&corpus)
		clipEncoded = corpEncoded
	}
	start := time.Now()
	decoded := clipEncoder.Decode(clipEncoded)
	duration := time.Since(start)
	tokenNumBytes := len(decoded)
	idxToStop := 229550
	t.Log(fmt.Sprintf("%v tokens into %v bytes over %v\n",
		len(*clipEncoded), tokenNumBytes, duration))
	for idx := range clipCorpus {
		if idx > idxToStop {
			break
		}

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
	for _, tokenizer := range []string{"clip-tokenizer", "gpt2-tokenizer", "pile-tokenizer"} {
		encoderFile := fmt.Sprintf("resources/data/%s/encoder.json", tokenizer)
		unitrimFile := fmt.Sprintf("resources/data/%s/unitrim.json", tokenizer)

		// make sure the files exist
		if _, err := os.Stat(encoderFile); os.IsNotExist(err) {
			t.Errorf("Could not find file %s\n", encoderFile)
		}
		if _, err := os.Stat(unitrimFile); os.IsNotExist(err) {
			t.Errorf("Could not find file %s\n", unitrimFile)
		}

		// read in the Encoder and unitrim files
		encoderBytes, err := os.ReadFile(encoderFile)
		// unmarshal the Encoder file
		var encoder map[string]Token
		err = json.Unmarshal(encoderBytes, &encoder)
		if err != nil {
			t.Errorf("Could not unmarshal Encoder file: %v\n", err)
		}

		// read in the unitrim file
		unitrimBytes, err := os.ReadFile(unitrimFile)
		// unmarshal the unitrim file
		var unitrim []int
		err = json.Unmarshal(unitrimBytes, &unitrim)
		if err != nil {
			t.Errorf("Could not unmarshal unitrim file: %v\n", err)
		}

		// get generated array for unitrim with the makeUnitrimArr function
		generatedArray := makeUnitrimArr(encoder)

		// check that the generated array is the same as the unitrim array
		fmt.Printf("Generated array length: %d, unitrim array length: %d\n", len(generatedArray), len(unitrim))
		if len(generatedArray) != len(unitrim) {
			t.Errorf("Generated array and unitrim array are not the same length\n")
		}

		for i := range generatedArray {
			if generatedArray[i] != unitrim[i] {
				fmt.Printf("Generated array: %v and unitrim array: %v at index %d are not the same\n", generatedArray[i], unitrim[i], i)
				fmt.Printf("mismatched unicode is: %c\n", rune(generatedArray[i]))
				t.Errorf("Generated array and unitrim array are not the same\n")
			}
		}

		fmt.Printf("Length and contents of generated array and unitrim array are the same\n")
	}
}

func TestLlamaEncoder_Encode(t *testing.T) {
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

func TestLlamaTwoEncoder_Encode(t *testing.T) {
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	llamaTokens := llama2Encoder.Encode(&testString)
	assert.Equal(t, llamaTokens, &Tokens{1576, 1701, 29916, 12500, 287, 975, 278, 447, 276, 29889, 13, 1576, 260, 4227, 280, 338, 8473, 1135, 278, 447, 276, 29889})
}

func TestLlamaTwoTokenizerDecode(t *testing.T) {
	outputString := "<s>The fox jumped over the hare.\nThe turtle is faster than the hare."
	llamaTokens := Tokens{1, 1576, 1701, 29916, 12500, 287, 975, 278, 447, 276, 29889, 13, 1576, 260, 4227, 280, 338, 8473, 1135, 278, 447, 276, 29889}
	output := llama2Encoder.Decode(&llamaTokens)
	assert.Equal(t, outputString, output)
}

func TestLlamaTwoEncodeDecode(t *testing.T) {
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	outputString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	llamaTokens := llama2Encoder.Encode(&testString)
	output := llama2Encoder.Decode(llamaTokens)
	assert.Equal(t, outputString, output)
}

func TestMistralEncoder_Encode(t *testing.T) {
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	mistralTokens := mistralEncoder.Encode(&testString)
	assert.Equal(t, mistralTokens, &Tokens{1, 415, 285, 1142, 14949, 754, 272, 295, 492, 28723, 13, 1014, 261, 3525, 291, 349, 9556, 821, 272, 295, 492, 28723})
}

func TestMistralTokenizerDecode(t *testing.T) {
	outputString := "<s> The fox jumped over the hare.\nThe turtle is faster than the hare."
	mistralTokens := Tokens{1, 415, 285, 1142, 14949, 754, 272, 295, 492, 28723, 13, 1014, 261, 3525, 291, 349, 9556, 821, 272, 295, 492, 28723}
	output := mistralEncoder.Decode(&mistralTokens)
	assert.Equal(t, outputString, output)
}

func TestMistralEncodeDecode(t *testing.T) {
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	outputString := "<s> The fox jumped over the hare.\nThe turtle is faster than the hare."
	mistralTokens := mistralEncoder.Encode(&testString)
	output := mistralEncoder.Decode(mistralTokens)
	assert.Equal(t, outputString, output)
}

func TestMistralEncodeDecodeFrankenstein(t *testing.T) {
	frankensteinCorpus := "resources/frankenstein.txt"
	frankensteinText, err := os.ReadFile(frankensteinCorpus)
	if err != nil {
		t.Errorf("Error reading Frankenstein corpus: %v", err)
	}
	frankensteinString := string(frankensteinText)
	mistralTokens := mistralEncoder.Encode(&frankensteinString)
	output := mistralEncoder.Decode(mistralTokens)
	assert.Equal(t, "<s> "+frankensteinString, output)
}

func TestReadTokenizerConfig(t *testing.T) {
	fmt.Println("Testing ReadTokenizerConfig")
	// json with eos, bos, pad as strings
	jsonStr := `{"eos_token": "TC", "bos_token": "TD", "pad_token": "TE"}` //cooresponds to 6669, 10989, 5428 in pythia vocab

	//download filler model
	modelId := "EleutherAI/pythia-70m"
	destPath := "./TestReadTokenizerConfig"
	destPathPTR := &destPath
	defer os.RemoveAll(destPath)
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

	// replace tokenizer_config.json with jsonStr
	tokenizerConfigPath := destPath + "/tokenizer_config.json"
	err := os.WriteFile(tokenizerConfigPath, []byte(jsonStr), 0644)
	if err != nil {
		t.Errorf("Error writing to tokenizer_config.json: %v", err)
	}

	// read tokenizer config by encoding a string
	encoder, err := NewEncoder(destPath)
	if err != nil {
		t.Errorf("Error creating encoder: %v", err)
	}

	// check that the tokens are correct
	assert.Equal(t, encoder.EosToken, Token(6669))
	assert.Equal(t, encoder.BosToken, Token(10989))
	assert.Equal(t, encoder.PadToken, Token(5428))

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good.")
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
	defer os.RemoveAll(destPath)
	_, rsrcErr := resources.ResolveResources(modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken)
	if rsrcErr != nil {
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
		t.Errorf("config.json does not exist")

	} else {
		t.Errorf("Error checking for config.json")
	}

	// Check for pytorch_model.bin
	modelPath := destPath + "/pytorch_model.bin"
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println("pytorch_model.bin exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("pytorch_model.bin does not exist")

	} else {
		t.Errorf("Error checking for pytorch_model.bin")
	}

	// Check for tokenizer.json
	tokenizerConfigPath := destPath + "/tokenizer.json"
	if _, err := os.Stat(tokenizerConfigPath); err == nil {
		fmt.Println("tokenizer.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("tokenizer.json does not exist")

	} else {
		t.Errorf("Error checking for tokenizer.json")
	}

	// Check for vocab.json
	vocabPath := destPath + "/vocab.json"
	if _, err := os.Stat(vocabPath); err == nil {
		fmt.Println("vocab.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("vocab.json does not exist")

	} else {
		t.Errorf("Error checking for vocab.json")
	}

	// Finish the test, allow defered cleanup
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
	defer os.RemoveAll(destPath)
	_, rsrcErr := resources.ResolveResources(modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken)
	if rsrcErr != nil {
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
		t.Errorf("config.json does not exist")

	} else {
		t.Errorf("Error checking for config.json")
	}

	// Check for pytorch_model.bin
	modelPath := destPath + "/pytorch_model.bin"
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println("pytorch_model.bin exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("pytorch_model.bin does not exist")

	} else {
		t.Errorf("Error checking for pytorch_model.bin")
	}

	// Check for tokenizer.json
	tokenizerConfigPath := destPath + "/tokenizer.json"
	if _, err := os.Stat(tokenizerConfigPath); err == nil {
		fmt.Println("tokenizer.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("tokenizer.json does not exist")

	} else {
		t.Errorf("Error checking for tokenizer.json")
	}

	// Check for vocab.json
	vocabPath := destPath + "/vocab.json"
	if _, err := os.Stat(vocabPath); err == nil {
		fmt.Println("vocab.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("vocab.json does not exist")

	} else {
		t.Errorf("Error checking for vocab.json")
	}

	// Finish the test, allow defered cleanup
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
	defer os.RemoveAll(destPath)
	_, rsrcErr := resources.ResolveResources(modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken)
	if rsrcErr != nil {
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
		t.Errorf("pytorch_model-00001-of-00002.bin does not exist")

	} else {
		t.Errorf("Error checking for pytorch_model-00001-of-00002.bin")
	}

	// Check for pytorch_model-00002-of-00002.bin
	model2Path := destPath + "/pytorch_model-00002-of-00002.bin"
	if _, err := os.Stat(model2Path); err == nil {
		fmt.Println("pytorch_model-00002-of-00002.bin exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("pytorch_model-00002-of-00002.bin does not exist")

	} else {
		t.Errorf("Error checking for pytorch_model-00002-of-00002.bin")
	}

	// Check for pytorch_model.bin.index.json
	shardconfigPath := destPath + "/pytorch_model.bin.index.json"
	if _, err := os.Stat(shardconfigPath); err == nil {
		fmt.Println("pytorch_model.bin.index.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("pytorch_model.bin.index.json does not exist")

	} else {
		t.Errorf("Error checking for pytorch_model.bin.index.json")
	}

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good.")

}

func TestModelDownloadLlama(t *testing.T) {
	// Pythia uses a slightly different file structure, where
	// the vocab.json and merges.txt files are stored in the
	// tokenizer.json file. We want to check if we are able to
	// download the model and extract the vocab.json and merges.txt
	modelId := "georgesung/llama2_7b_chat_uncensored"
	destPath := "./TestModelDownloadLlama"
	destPathPTR := &destPath
	defer os.RemoveAll(destPath)

	var rsrcType resources.ResourceType
	rsrcType = resources.RESOURCETYPE_TRANSFORMERS
	hfApiToken := os.Getenv("HF_API_TOKEN")
	os.MkdirAll(destPath, 0755)
	_, rsrcErr := resources.ResolveResources(modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken)
	if rsrcErr != nil {
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
		t.Errorf("config.json does not exist %s", err)

	} else {
		t.Errorf("Error checking for config.json %s", err)
	}

	// Check for pytorch_model.bin
	singleModelPattern := regexp.MustCompile(`pytorch_model\.bin$`)
	re, err := regexp.Compile(`-(\d+)-of-(\d+)\.bin$`)
	if err != nil {
		t.Errorf("Error compiling regex: %s", err)
	}

	//check all files in the directory against the pattern
	files, err := ioutil.ReadDir(destPath)
	if err != nil {
		t.Errorf("Error reading directory: %s", err)
	}
	found := false

	for _, file := range files {
		if singleModelPattern.MatchString(file.Name()) {
			found = true
			break
		}

		matches := re.FindStringSubmatch(file.Name())
		if matches != nil && len(matches) > 2 {
			if strings.Compare(matches[1], matches[2]) == 0 {
				found = true
				break
			}
		}
	}
	if !found {
		t.Errorf("pytorch_model.bin does not exist or was not found")
	}

	// Check for tokenizer.model
	tokenizerConfigPath := destPath + "/tokenizer.model"
	if _, err := os.Stat(tokenizerConfigPath); err == nil {
		fmt.Println("tokenizer.model exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("tokenizer.model does not exist. %s", err)

	} else {
		t.Errorf("Error checking for tokenizer.model. %s", err)
	}

	// Check for vocab.json
	vocabPath := destPath + "/vocab.json"
	if _, err := os.Stat(vocabPath); err == nil {
		fmt.Println("vocab.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("vocab.json does not exist. %s", err)

	} else {
		t.Errorf("Error checking for vocab.json. %s", err)
	}

	// Finish the test, allow defered cleanup
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
	defer os.RemoveAll(destPath)
	_, rsrcErr := resources.ResolveResources(modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken)
	if rsrcErr != nil {
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
		t.Errorf("config.json does not exist")

	} else {
		t.Errorf("Error checking for config.json")
	}

	// Check for pytorch_model.bin
	modelPath := destPath + "/pytorch_model.bin"
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println("pytorch_model.bin exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("pytorch_model.bin does not exist")

	} else {
		t.Errorf("Error checking for pytorch_model.bin")
	}

	// Check for vocab.json
	vocabPath := destPath + "/vocab.json"
	if _, err := os.Stat(vocabPath); err == nil {
		fmt.Println("vocab.json exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("vocab.json does not exist")

	} else {
		t.Errorf("Error checking for vocab.json")
	}

	// Check for merges.txt
	mergesPath := destPath + "/merges.txt"
	if _, err := os.Stat(mergesPath); err == nil {
		fmt.Println("merges.txt exists")

	} else if errors.Is(err, os.ErrNotExist) {
		t.Errorf("merges.txt does not exist")

	} else {
		t.Errorf("Error checking for merges.txt")
	}

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good (Fairseq Download).")
}

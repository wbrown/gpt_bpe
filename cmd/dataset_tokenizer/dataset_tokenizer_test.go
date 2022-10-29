package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/wbrown/gpt_bpe"
	"io"
	"log"
	"os"
	"testing"
	"time"
)

type SanitizerTest struct {
	Name     string
	Input    string
	Expected string
}

type SanitizerTests []SanitizerTest

var sanitizerTests = SanitizerTests{
	{"\\n handling",
		"\nfoobar\\n\n",
		"\nfoobar\n"},
	{"\\r handling",
		"\r\n\r\n",
		"\n"},
	{"Trailing spaces handling",
		"foobar  ",
		"foobar"},
	{"Extra spaces handling",
		"foo  bar",
		"foo bar"},
	{"Prefix spaces handling",
		" foo bar",
		"foo bar"},
	{"Colon with spaces handling",
		"foo : bar",
		"foo: bar"},
	{"Extra spaces with newlines",
		" foo \n   bar\nfoo ",
		"foo\nbar\nfoo"},
}

func BenchmarkSanitizeText(b *testing.B) {
	b.StopTimer()
	b.ResetTimer()
	path := "../../resources/frankenstein.txt"
	if testFile, err := os.Open(path); err != nil {
		b.Fail()
	} else {
		b.StartTimer()
		reader := CreateTextSanitizer(testFile)
		runes := make([]rune, 0)
		for {
			r, size, _ := reader.ReadRune()
			if size > 0 {
				runes = append(runes, r)
			} else {
				break
			}
		}
		b.StopTimer()
		b.Logf("%d runes read", len(runes))
	}
}

func BenchmarkStreamingEncode(b *testing.B) {
	b.StopTimer()
	b.ResetTimer()
	tokenizer := gpt_bpe.GPT2Encoder
	for i := 0; i < 5; i++ {
		if testFile, err := os.Open("../../resources/frankenstein." +
			"txt"); err != nil {

			b.Fail()
		} else {
			start := time.Now()
			b.StartTimer()
			nextChunk := tokenizer.StreamingEncode(bufio.NewReader(testFile))
			tokensCt := 0
			for {
				if chunk := nextChunk(2048); chunk == nil {
					break
				} else {
					tokensCt += len(*chunk)
				}
			}
			b.StopTimer()
			duration := time.Now().Sub(start)
			tokensPerSecond := float64(tokensCt) / duration.Seconds()
			b.Logf("%d tokens generated at %0.2f per second over %vms", tokensCt,
				tokensPerSecond, duration.Milliseconds())
		}
	}
}

func BenchmarkStreamingEncodeSanitize(b *testing.B) {
	b.StopTimer()
	b.ResetTimer()
	tokenizer := gpt_bpe.GPT2Encoder
	path := "../../resources/frankenstein.txt"
	if testFile, err := os.Open(path); err != nil {
		b.Fail()
	} else {
		start := time.Now()
		b.StartTimer()
		reader := CreateTextSanitizer(testFile)
		nextChunk := tokenizer.StreamingEncode(reader)
		tokensCt := 0
		for {
			if chunk := nextChunk(2048); chunk == nil {
				break
			} else {
				tokensCt += len(*chunk)
			}
		}
		b.StopTimer()
		tokensPerSecond := float64(tokensCt) / time.Now().Sub(start).Seconds()
		b.Logf("%d tokens generated at %0.2f per second", tokensCt,
			tokensPerSecond)
	}
}

func TestSanitizer(t *testing.T) {
	for testIdx := range sanitizerTests {
		input := sanitizerTests[testIdx].Input
		output := SanitizeText(input)
		assert.Equal(t, sanitizerTests[testIdx].Expected, output)
	}
}

func TestSanitizedRuneReader_ReadRune(t *testing.T) {
	for testIdx := range sanitizerTests {
		input := sanitizerTests[testIdx].Input
		reader := CreateTextSanitizer(bytes.NewBufferString(input))
		runes := make([]rune, 0)
		for {
			r, size, _ := reader.ReadRune()
			if size > 0 {
				runes = append(runes, r)
			} else {
				break
			}
		}
		output := string(runes)
		assert.Equal(t, sanitizerTests[testIdx].Expected, output)
	}
}

func TestSampling40(t *testing.T) {
	all1 := 0
	all2 := 0

	textsTokenizer := NewTextsTokenizer()
	textsTokenizer.ContextSize = 2048
	textsTokenizer.TokenizerId = "gpt2"
	textsTokenizer.EndOfText = ""
	textsTokenizer.PadToken = ""
	textsTokenizer.Boundary = "\n"
	textsTokenizer.Unitrim = true
	textsTokenizer.BoundaryBegin = false

	inputDir := "../../resources"
	reorderPaths := ""
	sampling := 100
	outputFile := "base.chunk"

	if _, tokErr := textsTokenizer.InitTokenizer(); tokErr != nil {
		log.Fatal(tokErr)
	}

	if nextText, err := ReadTexts(inputDir, false,
		reorderPaths); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts, tokErr := textsTokenizer.TokenizeTexts(
			nextText)
		if tokErr != nil {
			log.Fatal(tokErr)
		}

		var enc *gpt_bpe.GPTEncoder
		// *showContexts = true

		total, writeErr := WriteContexts(outputFile, contexts, enc, sampling, reorderPaths)
		all1 += total
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", total,
			duration, float64(total)/duration)
	}

	textsTokenizer2 := NewTextsTokenizer()
	textsTokenizer2.ContextSize = 2048
	textsTokenizer2.TokenizerId = "gpt2"
	textsTokenizer2.EndOfText = ""
	textsTokenizer2.PadToken = ""
	textsTokenizer2.Boundary = "\n"
	textsTokenizer2.Unitrim = true
	textsTokenizer2.BoundaryBegin = false

	inputDir = "../../resources"
	reorderPaths = ""
	sampling = 40
	outputFile = "samp40.chunk"

	if _, tokErr := textsTokenizer.InitTokenizer(); tokErr != nil {
		log.Fatal(tokErr)
	}

	if nextText, err := ReadTexts(inputDir, false,
		reorderPaths); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts, tokErr := textsTokenizer.TokenizeTexts(
			nextText)
		if tokErr != nil {
			log.Fatal(tokErr)
		}
		var enc *gpt_bpe.GPTEncoder
		// *showContexts = true

		total2, writeErr := WriteContexts(outputFile, contexts, enc, sampling, reorderPaths)
		all2 += total2
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", total2,
			duration, float64(total2)/duration)
	}
	percent := (float64(all2) / float64(all1)) * 100
	log.Printf("Sampling 100 produced %d Tokens, Sampling 40 produced %d tokens, this is roughly %f %%", all1, all2, percent)
	if percent > 55 || percent < 25 {
		log.Printf("Percent does not match ~40%% (25-55), found to be %f", percent)
		t.Fail()
	}
}

func TestShuffle(t *testing.T) {
	all1 := 0
	all2 := 0

	textsTokenizer := NewTextsTokenizer()
	textsTokenizer.ContextSize = 2048
	textsTokenizer.TokenizerId = "gpt2"
	textsTokenizer.EndOfText = ""
	textsTokenizer.PadToken = ""
	textsTokenizer.Boundary = "\n"
	textsTokenizer.Unitrim = true
	textsTokenizer.BoundaryBegin = false

	inputDir := "../../resources"
	reorderPaths := "size_ascending"
	sampling := 100
	outputFile := "noshuffle.chunk"

	if _, tokErr := textsTokenizer.InitTokenizer(); tokErr != nil {
		log.Fatal(tokErr)
	}

	if nextText, err := ReadTexts(inputDir, false,
		reorderPaths); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts, tokErr := textsTokenizer.TokenizeTexts(
			nextText)
		if tokErr != nil {
			log.Fatal(tokErr)
		}
		var enc *gpt_bpe.GPTEncoder
		// *showContexts = true

		total, writeErr := WriteContexts(outputFile, contexts, enc, sampling, reorderPaths)
		all1 += total
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", total,
			duration, float64(total)/duration)
	}

	textsTokenizer2 := NewTextsTokenizer()
	textsTokenizer2.ContextSize = 2048
	textsTokenizer2.TokenizerId = "gpt2"
	textsTokenizer2.EndOfText = ""
	textsTokenizer2.PadToken = ""
	textsTokenizer2.Boundary = "\n"
	textsTokenizer2.Unitrim = true
	textsTokenizer2.BoundaryBegin = false

	inputDir = "../../resources"
	reorderPaths = "shuffle"
	sampling = 100
	outputFile = "shuffle.chunk"

	if _, tokErr := textsTokenizer.InitTokenizer(); tokErr != nil {
		log.Fatal(tokErr)
	}

	if nextText, err := ReadTexts(inputDir, false,
		reorderPaths); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts, tokErr := textsTokenizer.TokenizeTexts(
			nextText)
		if tokErr != nil {
			log.Fatal(tokErr)
		}
		var enc *gpt_bpe.GPTEncoder
		// *showContexts = true

		total2, writeErr := WriteContexts(outputFile, contexts, enc, sampling, reorderPaths)
		all2 += total2
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", total2,
			duration, float64(total2)/duration)
	}
	percent := (float64(all2) / float64(all1)) * 100
	log.Printf("NoShuffle produced %d Tokens, Shuffle produced %d tokens, this is roughly %f %%", all1, all2, percent)
	if percent != 100 {
		log.Printf("Percent does not match ~100%%, found to be %f", percent)
		t.Fail()
	}

	f, err := os.Open("noshuffle.chunk")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	f2, err2 := os.Open("shuffle.chunk")
	if err2 != nil {
		log.Fatal(err2)
	}
	defer f2.Close()

	//chunk by chunk verification of shuffle
	f.Seek(0, 0)
	var verifymap = make(map[string]bool)
	buffer := make([]byte, 4096)
	for {
		bytesread, err := f.Read(buffer)
		if err != nil {
			if err != io.EOF {
				fmt.Println(err)
			}
			break
		}
		hash := sha256.Sum256(buffer[0 : bytesread-1])
		sha := hex.EncodeToString(hash[:])
		verifymap[sha] = false
	}
	f2.Seek(0, 0)
	buffer2 := make([]byte, 4096)
	for {
		bytesread, err := f2.Read(buffer2)
		if err != nil {
			if err != io.EOF {
				fmt.Println(err)
			}
			break
		}
		hash2 := sha256.Sum256(buffer2[0 : bytesread-1])
		sha2 := hex.EncodeToString(hash2[:])
		verifymap[sha2] = true
	}

	for _, value := range verifymap {
		//fmt.Printf("%s value is %v\n", key, value)
		if value == false {
			fmt.Printf("Found failing chunk (chunk didn't match)")
			t.Fail()
		}
	}

	fmt.Printf("Using Chunk by chunk hashing, shuffle found to be working as intended!! \n")

}

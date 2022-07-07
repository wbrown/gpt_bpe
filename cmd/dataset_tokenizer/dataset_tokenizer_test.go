package main

import (
	"bufio"
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/wbrown/gpt_bpe"
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
		if testFile, err := os.Open("../../all." +
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
			tokensPerSecond := float64(tokensCt) / time.Now().Sub(start).Seconds()
			b.Logf("%d tokens generated at %0.2f per second", tokensCt,
				tokensPerSecond)
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

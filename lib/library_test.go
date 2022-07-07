package main

import (
	"fmt"
	"os"
	"testing"
)

func ReadTestFile(path string) ([]byte, error) {
	if testFile, err := os.Open(path); err != nil {
		return nil, err
	} else {
		fileinfo, statErr := testFile.Stat()
		if statErr != nil {
			fmt.Println(err)
			return nil, statErr
		}
		filesize := fileinfo.Size()
		buffer := make([]byte, filesize)

		_, readErr := testFile.Read(buffer)
		if readErr != nil {
			fmt.Println(err)
			return nil, readErr
		}

		return buffer, nil
	}
}

func BenchmarkTokenize(b *testing.B) {
	b.StopTimer()
	b.ResetTimer()
	vocab := "gpt2-tokenizer"
	wrapInitTokenizer(vocab)
	path := "../resources/frankenstein.txt"
	if corpus, err := ReadTestFile(path); err != nil {
		fmt.Printf("%v\n", err)
		b.Fail()
	} else {
		b.StartTimer()
		duration, numTokens := testBuffer(vocab, corpus)
		b.StopTimer()
		tokensPerSecond := float64(numTokens) / duration.Seconds()
		b.Logf("%d tokens generated at %0.2f per second over %vms",
			numTokens, tokensPerSecond, duration.Milliseconds())
	}
}

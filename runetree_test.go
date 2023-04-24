package gpt_bpe

import (
	"io"
	"strings"
	"testing"
)

func TestRuneNode_String(t *testing.T) {
	print(nerdstashEncoder.specialsTree.String())
}

func TestRuneMatch(t *testing.T) {
	nerdstashEncoder.createRuneTree()
	s := "// TypeScript Version: 2.9"
	rr := io.RuneReader(strings.NewReader(s))
	nextWord := nerdstashEncoder.WordSplitter(rr)
	for {
		word := nextWord()
		if word == nil {
			break
		}
		t.Log(*word)
	}
}

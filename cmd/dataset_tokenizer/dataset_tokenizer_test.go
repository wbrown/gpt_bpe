package main

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"testing"
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

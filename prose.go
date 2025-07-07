//go:build !wasip1 && !js

package gpt_bpe

import (
	"strings"
	"unicode"

	"github.com/jdkato/prose/v2"
)

func (encoder *GPTEncoder) TrimIncompleteSentence(tokens *Tokens) (
	*Tokens,
	error,
) {
	trimmed := make(Tokens, 0)
	doc, err := prose.NewDocument(
		encoder.Decode(tokens),
		prose.WithTagging(false),
		prose.WithExtraction(false),
		prose.WithTokenization(false),
	)
	if err != nil {
		return &trimmed, err
	}
	firstSentences := doc.Sentences()
	sentences := make([]string, 0)
	for _, sentence := range firstSentences {
		newSentences := encoder.puncPat.Split(sentence.Text, -1)
		sentences = append(sentences, newSentences...)
	}
	lastSentence := sentences[len(sentences)-1]
	var last rune
	for _, r := range lastSentence {
		if unicode.IsSpace(r) {
			continue
		}
		last = r
	}
	var text = doc.Text
	if !unicode.IsPunct(last) {
		trimPos := strings.LastIndex(text, lastSentence)
		if trimPos >= 1 {
			text = doc.Text[:trimPos-1]
		}
	}
	text = strings.TrimSpace(text)
	if float32(len(text)) < float32(len(doc.Text))*0.8 {
		return tokens, nil
	}
	encoded := encoder.Encode(&text)
	return encoded, nil
}

func (encoder *GPTEncoder) TrimSentences(
	tokens *Tokens,
	direction TrimDirection,
	limit uint,
) (*Tokens, error) {
	var err error
	trimmed := make(Tokens, 0)
	if uint(len(*tokens)) <= limit {
		return tokens, err
	} else if direction == TrimNone {
		return &trimmed, err
	}
	doc, err := prose.NewDocument(
		encoder.Decode(tokens),
		prose.WithTagging(false),
		prose.WithExtraction(false),
		prose.WithTokenization(false),
	)
	if err != nil {
		return &trimmed, err
	}
	sentences := doc.Sentences()
	var start, end, step, idx int
	var textBegin, textEnd int
	var sentenceIdx, lastSentence int
	switch direction {
	case TrimTop:
		start = len(sentences) - 1
		end = -1
		step = -1
		textBegin = 0
		textEnd = len(doc.Text)
	case TrimBottom:
		start = 0
		end = len(sentences)
		step = 1
		textBegin = 0
		textEnd = len(doc.Text)
	default:
		return &trimmed, err
	}
	for idx = start; idx != end; idx += step {
		sentence := sentences[idx].Text
		switch direction {
		case TrimTop:
			sentenceIdx = strings.LastIndex(
				doc.Text[textBegin:],
				sentence,
			) + textBegin
			if sentenceIdx > 0 && sentenceIdx < len(doc.Text) &&
				unicode.IsSpace(rune(doc.Text[sentenceIdx])) {
				sentenceIdx -= 1
			}
			toTokenize := doc.Text[sentenceIdx:]
			tokCt := uint(len(*(encoder.Encode(&toTokenize))))
			if tokCt >= limit {
				toEncode := doc.Text[textEnd:]
				return encoder.Encode(&toEncode), err
			}
			textEnd = sentenceIdx - 1
		case TrimBottom:
			sentenceIdx = strings.Index(
				doc.Text[textBegin:textEnd],
				sentence,
			) + textBegin
			sentenceEnd := sentenceIdx + len(sentence)
			if sentenceEnd < textEnd &&
				doc.Text[sentenceEnd:sentenceEnd+1] == "\n" {
				sentenceEnd += 1
			}
			toTokenize := doc.Text[0:sentenceEnd]
			tokCt := uint(len(*(encoder.Encode(&toTokenize))))
			if tokCt >= limit {
				toEncode := doc.Text[0:lastSentence]
				return encoder.Encode(&toEncode), err
			}
			lastSentence = sentenceEnd
			textBegin += len(sentence)
		}
	}
	return &trimmed, err
}

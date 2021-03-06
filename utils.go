package gpt_bpe

import (
	"bytes"
	"encoding/binary"
	"github.com/jdkato/prose/v2"
	"strings"
	"unicode"
)

type TrimDirection uint

const (
	TrimTop    TrimDirection = iota
	TrimBottom TrimDirection = iota
	TrimNone   TrimDirection = iota
)

const (
	TokenSize = 2
)

func (tokens *Tokens) ToBin() *[]byte {
	buf := bytes.NewBuffer(make([]byte, 0, len(*tokens)*TokenSize))
	for idx := range *tokens {
		bs := (*tokens)[idx]
		binary.Write(buf, binary.LittleEndian, bs)
	}
	byt := buf.Bytes()
	return &byt
}

func TokensFromBin(bin *[]byte) *Tokens {
	tokens := make(Tokens, 0)
	buf := bytes.NewReader(*bin)
	for {
		var token Token
		if err := binary.Read(buf, binary.LittleEndian, &token); err != nil {
			break
		}
		tokens = append(tokens, token)
	}
	return &tokens
}

func (encoder GPTEncoder) TrimNewlines(tokens *Tokens, direction TrimDirection,
	limit uint) (*Tokens, error) {
	var err error
	trimmed := make(Tokens, 0)
	if uint(len(*tokens)) <= limit {
		return tokens, err
	} else if direction == TrimNone {
		return &trimmed, err
	}
	lines := strings.Split(encoder.Decode(tokens), "\n")
	var start, end, step, idx int
	switch direction {
	case TrimTop:
		start = len(lines) - 1
		end = -1
		step = -1
	case TrimBottom:
		start = 0
		end = len(lines)
		step = 1
	}
	accTokens := make(Tokens, 0)
	for idx = start; idx != end; idx += step {
		line := lines[idx]
		switch direction {
		case TrimTop:
			line = "\n" + line
		case TrimBottom:
			line = line + "\n"
		}
		newTokens := encoder.Encode(&line)
		if len(*newTokens)+len(accTokens) > int(limit) {
			return &accTokens, err
		} else {
			switch direction {
			case TrimTop:
				accTokens = append(*newTokens, accTokens...)
			case TrimBottom:
				accTokens = append(accTokens, *newTokens...)
			}
		}
	}
	return &accTokens, err
}

func (encoder GPTEncoder) TrimIncompleteSentence(tokens *Tokens) (*Tokens, error) {
	trimmed := make(Tokens, 0)
	doc, err := prose.NewDocument(encoder.Decode(tokens),
		prose.WithTagging(false),
		prose.WithExtraction(false),
		prose.WithTokenization(false))
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

func (encoder GPTEncoder) AlignAndSizeTokens(tokens *Tokens,
	desiredLength int) (alignedTokens Tokens, endAt int) {
	chunk := (*tokens)[0:desiredLength]
	// We trim to valid tokens, as we don't want partials
	// that are truncated multi-tokens.
	trimmed := encoder.TrimTokens(&chunk)
	trimmedLength := len(*trimmed)
	isTrimmed := len(*trimmed) != len(chunk)
	chunk = *trimmed
	idx := trimmedLength

	// We do a decode and reencode pass, as this can affect
	// the size after a trim.
	if isTrimmed {
		decodedChunk := encoder.Decode(&chunk)
		reencodedChunk := encoder.Encode(&decodedChunk)
		chunk = *reencodedChunk
		// See if there's any change in size that causes it to
		// be smaller than the `desiredLength`.
		roundtripRemainder := desiredLength - len(chunk)
		if roundtripRemainder > 0 {
			addlEnd := idx + roundtripRemainder
			addlTokens := (*tokens)[idx:addlEnd]
			trimmedAddl := encoder.TrimTokens(&addlTokens)
			chunk = append(chunk, *trimmedAddl...)
			idx += len(*trimmedAddl)
			// Another decode/re-encode pass.
			decodedChunk = encoder.Decode(&chunk)
			reencodedChunk = encoder.Encode(&decodedChunk)
			// Loop, dropping tokens one by one until we have
			// valid tokens and we fit within `contextSize`.
			for {
				chunk = *reencodedChunk
				if len(chunk) <= desiredLength &&
					encoder.TokensReady(&chunk) {
					break
				}
				chunk = chunk[:len(chunk)-1]
				idx -= 1
				decodedChunk = encoder.Decode(&chunk)
				reencodedChunk = encoder.Encode(&decodedChunk)
			}
		}
	}

	return chunk, idx
}

func (encoder GPTEncoder) TrimSentences(tokens *Tokens, direction TrimDirection,
	limit uint) (*Tokens, error) {
	var err error
	trimmed := make(Tokens, 0)
	if uint(len(*tokens)) <= limit {
		return tokens, err
	} else if direction == TrimNone {
		return &trimmed, err
	}
	doc, err := prose.NewDocument(encoder.Decode(tokens),
		prose.WithTagging(false),
		prose.WithExtraction(false),
		prose.WithTokenization(false))
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
			sentenceIdx = strings.LastIndex(doc.Text[textBegin:],
				sentence) + textBegin
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
			sentenceIdx = strings.Index(doc.Text[textBegin:textEnd],
				sentence) + textBegin
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

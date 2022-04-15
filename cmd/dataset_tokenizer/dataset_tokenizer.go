package main

import (
	"errors"
	"flag"
	"fmt"
	"github.com/wbrown/gpt_bpe"
	"github.com/yargevad/filepathx"
	"log"
	"os"
	"time"
)

var tokenizers map[string]*gpt_bpe.GPTEncoder

type TextsIterator func() *string

// ReadTexts
// Consumes a directory path and recursively scans for `.txt` files, producing
// a TextsIterator function that yields the text file contents.
func ReadTexts(dirPath string) (TextsIterator, error) {
	matches, err := filepathx.Glob(dirPath + "/**/*.txt")
	if err != nil {
		return nil, err
	}
	matchIdx := 0
	numMatches := len(matches)
	if numMatches == 0 {
		return nil, errors.New(fmt.Sprintf(
			"%s does not contain any .txt files", dirPath))
	}
	return func() *string {
		if matchIdx < numMatches {
			path := matches[matchIdx]
			if textBytes, readErr := os.ReadFile(path); readErr != nil {
				log.Fatal(readErr)
			} else {
				log.Print("Reading ", path)
				matchIdx += 1
				text := string(textBytes)
				return &text
			}
		}
		return nil
	}, nil
}

type TextsTokenizer struct {
	TokenizerId string
	ContextSize int
	Boundary    string
	PadToken    string
	EndOfText   string
	Unitrim     bool
}

func NewTextsTokenizer() TextsTokenizer {
	return TextsTokenizer{
		"gpt2",
		2048,
		"\n",
		"<|endoftext|>",
		"<|endoftext|>",
		true,
	}
}

func getAndCheckToken(t *gpt_bpe.GPTEncoder, s string,
	id string) (gpt_bpe.Token, error) {
	token := t.Get(s)
	if token == nil {
		tokens := t.Encode(&s)
		if len(*tokens) != 1 {
			return 0, errors.New(fmt.Sprintf(
				"'%s' is not a valid token for %s", s, id))
		} else {
			return (*tokens)[0], nil
		}
	} else {
		return *token, nil
	}
}

type ContextsIterator func() *gpt_bpe.Tokens

// TokenizeTexts
// Consumes a TextsIterator and produces a ContextsIterator iterator function
// that returns tokenized contexts that are fixed and padded out to
// `contextSize`.
func (tt TextsTokenizer) TokenizeTexts(
	nextText TextsIterator) (ContextsIterator, error) {
	tokenizerPtr, ok := tokenizers[tt.TokenizerId]
	if !ok {
		var tokErr error
		tokenizerPtr, tokErr = gpt_bpe.NewEncoder(tt.TokenizerId)
		if tokErr != nil {
			return nil, tokErr
		}
	}
	tokenizer := *tokenizerPtr
	padToken, padErr := getAndCheckToken(&tokenizer, tt.PadToken,
		"PadToken")
	if padErr != nil {
		return nil, padErr
	}
	endOfText, eotErr := getAndCheckToken(&tokenizer, tt.EndOfText,
		"EndOfText")
	if eotErr != nil {
		return nil, eotErr
	}
	var boundary gpt_bpe.Token
	if tt.Boundary == "" {
		boundary = 65535
	} else {
		var boundaryErr error
		boundary, boundaryErr = getAndCheckToken(&tokenizer, tt.Boundary,
			"Boundary")
		if boundaryErr != nil {
			return nil, boundaryErr
		}
	}
	contextSize := tt.ContextSize
	doUnitrim := tt.Unitrim

	prior := make(gpt_bpe.Tokens, 0)

	var tokens *gpt_bpe.Tokens
	var text *string
	var done bool
	var numTokens, idx, begin, boundaryIdx int

	// Consume texts from `nextText()` and tokenize as a `goroutine`.
	tokenizedTexts := make(chan *gpt_bpe.Tokens, 1)
	nextTokenized := func() {
		for {
			text = nextText()
			if text != nil {
				tokenizedTexts <- tokenizer.Encode(text)
			} else {
				close(tokenizedTexts)
				break
			}
		}
	}
	go nextTokenized()

	// Consumes tokenized texts and resets closured states for token blocks.
	nextInput := func() {
		var more bool
		tokens, more = <-tokenizedTexts
		idx = 0
		begin = 0
		boundaryIdx = 0
		if more {
			done = false
			numTokens = len(*tokens)
		} else {
			done = true
		}
	}

	// Prime the pump by initializing the states.
	nextInput()

	// Return an iterator function that returns token chunks that are always
	// `contextSize` tokens.
	nextContext := func() *gpt_bpe.Tokens {
		// Loop until we get a full token chunk.
		for {
			if done {
				// We're completely done and have no more token chunks to
				// return, so we flush out and pad the last chunk.
				if len(prior) > 0 {
					padSize := contextSize - len(prior)
					if padSize > 0 {
						for padIdx := 0; padIdx < padSize; padIdx += 1 {
							prior = append(prior, padToken)
						}
					}
					ret := prior
					prior = make(gpt_bpe.Tokens, 0)
					return &ret
				} else {
					// We just flushed the last token chunk, and have no more
					// so we return `nil`.
					return nil
				}
			}
			// Iterate until we reach the end of this text's tokens.
			for idx < numTokens {
				token := (*tokens)[idx]
				// Mark the position of the last `boundary` token we've seen.
				if token == boundary {
					boundaryIdx = idx
				}
				// Determine if we're at least `contextSize` yet, and if so
				// we do the finalization of this context.
				if idx-begin+len(prior) >= contextSize {
					chunk := (*tokens)[begin:idx]
					// If we have `prior` from a prior text, we prepend the
					// beginning of this text.
					if len(prior) > 0 {
						prior = append(prior, chunk...)
						chunk = prior
						prior = make(gpt_bpe.Tokens, 0)
					}
					var isTrimmed bool
					if doUnitrim {
						// We trim to valid tokens, as we don't want partials
						// that are truncated multi-tokens.
						trimmed := tokenizer.TrimTokens(&chunk)
						trimmedLength := len(*trimmed)
						isTrimmed = len(*trimmed) != len(chunk)
						chunk = *trimmed
						idx = begin + trimmedLength
					}
					// We do a decode and reencode pass, as this can affect
					// the size after a trim.
					if isTrimmed {
						decodedChunk := tokenizer.Decode(&chunk)
						reencodedChunk := tokenizer.Encode(&decodedChunk)
						chunk = *reencodedChunk
						// See if there's any change in size that causes it to
						// be smaller than the `contextSize`.
						roundtripRemainder := contextSize - len(chunk)
						if roundtripRemainder > 0 {
							addlEnd := idx + roundtripRemainder
							addlTokens := (*tokens)[idx:addlEnd]
							trimmedAddl := tokenizer.TrimTokens(&addlTokens)
							chunk = append(chunk, *trimmedAddl...)
							idx += len(*trimmedAddl)
							// Another decode/re-encode pass.
							decodedChunk = tokenizer.Decode(&chunk)
							reencodedChunk = tokenizer.Encode(&decodedChunk)
							// Loop, dropping tokens one by one until we have
							// valid tokens and we fit within `contextSize`.
							for {
								chunk = *reencodedChunk
								if len(chunk) <= contextSize &&
									tokenizer.TokensReady(&chunk) {
									break
								}
								chunk = chunk[:len(chunk)-1]
								idx -= 1
								decodedChunk = tokenizer.Decode(&chunk)
								reencodedChunk = tokenizer.Encode(&decodedChunk)
							}
						}
					}
					// If we have less than `contextSize`, we need to pad out
					// the tokens in this context.
					padSize := contextSize - len(chunk)
					if padSize > 0 {
						for padIdx := 0; padIdx < padSize; padIdx += 1 {
							chunk = append(chunk, padToken)
						}
					}
					// We had a boundary token in our last context, so set the
					// `idx` to one past the boundary token. This effectively
					// copies the chunk from that point on into the next
					// returned context.
					if boundaryIdx > 0 {
						if idx-boundaryIdx+1 <= contextSize {
							idx = boundaryIdx + 1
						}
						boundaryIdx = 0
					}
					// Reset the `begin` offsets, move idx, to set up the
					// state for the next invocation of this function.
					begin = idx
					idx += 1
					return &chunk
				}
				idx += 1
			}
			// If we've reached this, and have less tokens than `numTokens`,
			// we append `<|endoftext|>` token and carry it over to the next
			// text's batch in `prior`.
			if begin < numTokens {
				prior = append(prior, (*tokens)[begin:]...)
				if len(prior) < contextSize {
					prior = append(prior, endOfText)
				}
				if len(prior) == contextSize {
					chunk := &prior
					prior = make(gpt_bpe.Tokens, 0)
					nextInput()
					return chunk
				}
			}
			// Fetch the next text's tokens.
			nextInput()
		}
	}
	return nextContext, nil
}

// WriteContexts
// Consumes a ContextsIterator function and serializes the contexts to an
// aligned binary file.
func WriteContexts(outPath string, nextContext ContextsIterator) (int, error) {
	totalTokens := 0
	outFile, err := os.OpenFile(outPath, os.O_TRUNC|os.O_WRONLY|os.O_CREATE,
		755)
	if err != nil {
		return 0, err
	}
	for {
		context := nextContext()
		if context == nil {
			break
		}

		binContext := context.ToBin()
		if _, writeErr := outFile.Write(*binContext); writeErr != nil {
			return totalTokens, writeErr
		}
		totalTokens += len(*context)
	}
	return totalTokens, nil
}

func init() {
	tokenizers = make(map[string]*gpt_bpe.GPTEncoder, 0)
	tokenizers["gpt2"], _ = gpt_bpe.NewEncoder("gpt2")
	tokenizers["pile"], _ = gpt_bpe.NewEncoder("pile")
}

func main() {
	tokenizerId := flag.String("tokenizer", "gpt2",
		"tokenizer to use [gpt2, pile]")
	contextSize := flag.Int("context", 2048, "context size")
	endOfText := flag.String("eot", "<|endoftext|>",
		"end of text token to split texts")
	padToken := flag.String("pad", "<|endoftext|>",
		"pad token to pad out contexts, can be <|padding|>")
	boundaryToken := flag.String("boundary", "\n",
		"boundary token to split contexts on")
	outputFile := flag.String("output", "tokenized.chunk",
		"tokenized output file")
	inputDir := flag.String("input", "",
		"input directory")
	unitrimBool := flag.Bool("unitrim", true,
		"trim contexts to valid unicode")
	flag.Parse()
	if *inputDir == "" {
		flag.Usage()
		log.Fatal("Must provide -input for directory source")
	}

	textsTokenizer := NewTextsTokenizer()
	textsTokenizer.ContextSize = *contextSize
	textsTokenizer.TokenizerId = *tokenizerId
	textsTokenizer.EndOfText = *endOfText
	textsTokenizer.PadToken = *padToken
	textsTokenizer.Boundary = *boundaryToken
	textsTokenizer.Unitrim = *unitrimBool

	if nextText, err := ReadTexts(*inputDir); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts, tokErr := textsTokenizer.TokenizeTexts(
			nextText)
		if tokErr != nil {
			log.Fatal(tokErr)
		}
		total, writeErr := WriteContexts(*outputFile, contexts)
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		log.Printf("%d tokens in %0.2fs", total,
			time.Now().Sub(begin).Seconds())
	}
}

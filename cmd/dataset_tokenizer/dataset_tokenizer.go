package main

import (
	"github.com/wbrown/gpt_bpe"
	"github.com/yargevad/filepathx"
	"log"
	"os"
	"time"
)

func readTexts(dirPath string) (func() *string, error) {
	matches, err := filepathx.Glob(dirPath + "/**/*.txt")
	if err != nil {
		return nil, err
	}
	matchIdx := 0
	numMatches := len(matches)
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

// Consumes `nextText`, an iterator function that returns corpus texts, and
// returns a closured `nextContext` iterator function that returns tokenized
// contexts that are fixed to `contextSize` in length. The contexts overlap
// at boundaries set to `boundary` tokens.
func tokenizeTexts(nextText func() *string, contextSize int,
	boundary gpt_bpe.Token) func() *gpt_bpe.Tokens {
	tokenizer := gpt_bpe.NewGPT2Encoder()
	// endoftext := "<|endoftext|>"
	padToken := gpt_bpe.Token(50256)
	endOfText := gpt_bpe.Token(50256)
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
		var ok bool
		tokens, ok = <-tokenizedTexts
		idx = 0
		begin = 0
		boundaryIdx = 0
		if ok {
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
	return func() *gpt_bpe.Tokens {
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
					// We trim to valid tokens, as we don't want partials that
					// are truncated multi-tokens.
					trimmed := tokenizer.TrimTokens(&chunk)
					trimmedLength := len(*trimmed)
					idx = begin + trimmedLength
					// If we have `prior` from a prior text, we append the
					// beginning of this text.
					if len(prior) > 0 {
						prior = append(prior, *trimmed...)
						chunk = prior
						prior = make(gpt_bpe.Tokens, 0)
					}
					// We do a decode and reencode pass, as this can affect
					// the size after a trim.
					decodedChunk := tokenizer.Decode(&chunk)
					reencodedChunk := tokenizer.Encode(&decodedChunk)
					chunk = *reencodedChunk
					// See if there's any change in size that causes it to
					// be smaller than the `contextSize`.
					roundtripRemainder := contextSize - len(chunk)
					if roundtripRemainder > 0 {
						addlTokens := (*tokens)[idx : idx+roundtripRemainder]
						trimmedAddl := tokenizer.TrimTokens(&addlTokens)
						chunk = append(chunk, *trimmedAddl...)
						idx += len(*trimmedAddl)
						// Another decode/re-encode pass.
						decodedChunk = tokenizer.Decode(&chunk)
						reencodedChunk = tokenizer.Encode(&decodedChunk)
						// Loop, dropping tokens one by one until we have valid
						// tokens and we fit within `contextSize`.
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
					// If we have less than `contextSize`, we need to pad out
					// the tokens in this context.
					padSize := contextSize - len(chunk)
					if padSize > 0 {
						for padIdx := 0; padIdx <= padSize; padIdx += 1 {
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
}

// Consumes a closured `nextContext` iterator and produces a binary file that
// contains serialized and aligned contexts.
func writeContexts(outPath string, nextContext func() *gpt_bpe.Tokens) int {
	totalTokens := 0
	outFile, err := os.OpenFile(outPath, os.O_TRUNC|os.O_WRONLY|os.O_CREATE,
		755)
	if err != nil {
		log.Fatal(err)
	}
	for {
		context := nextContext()
		if context == nil {
			break
		}

		binContext := context.ToBin()
		if _, writeErr := outFile.Write(*binContext); writeErr != nil {
			log.Fatal(writeErr)
		}
		totalTokens += len(*context)
	}
	return totalTokens
}

func main() {
	dir := os.Args[1]
	output := os.Args[2]

	if nextText, err := readTexts(dir); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		total := writeContexts(output,
			tokenizeTexts(nextText, 2048, gpt_bpe.Token(198)))
		log.Printf("%d tokens in %0.2fs", total,
			time.Now().Sub(begin).Seconds())
	}
}

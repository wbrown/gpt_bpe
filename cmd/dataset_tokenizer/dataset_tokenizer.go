package main

import (
	"errors"
	"flag"
	"fmt"
	"github.com/wbrown/gpt_bpe"
	"github.com/yargevad/filepathx"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

var tokenizers map[string]*gpt_bpe.GPTEncoder

type TextsIterator func() *string

func SanitizeText(text string) string {
	extraNewLines := regexp.MustCompile("(?m)\n+")
	extraWhiteSpace := regexp.MustCompile("[[:space:]]+")

	text = strings.ReplaceAll(text, "\\n", "\n")
	text = strings.ReplaceAll(text, "\r", "")
	text = extraNewLines.ReplaceAllString(text, "\n")
	lines := strings.Split(text, "\n")
	for lineIdx := range lines {
		line := lines[lineIdx]
		line = strings.ReplaceAll(line, "\t", " ")
		line = strings.ReplaceAll(line, " :", ":")
		line = extraWhiteSpace.ReplaceAllString(line, " ")
		line = strings.TrimSpace(line)
		lines[lineIdx] = line
	}
	return strings.Join(lines, "\n")
}

// GlobTexts
// Given a directory path, recursively finds all `.txt` files.
func GlobTexts(dirPath string) (textPaths []string, err error) {
	textPaths, err = filepathx.Glob(dirPath + "/**/*.txt")
	if err != nil {
		return nil, err
	}
	numMatches := len(textPaths)
	if numMatches == 0 {
		return nil, errors.New(fmt.Sprintf(
			"%s does not contain any .txt files", dirPath))
	}
	return textPaths, nil
}

func FindNewestPath(paths *[]string) (path *string, newest *time.Time,
	err error) {
	for matchIdx := range *paths {
		currPath := (*paths)[matchIdx]
		if stat, statErr := os.Stat(currPath); statErr != nil {
			return nil, nil, statErr
		} else if newest == nil || newest.Before(stat.ModTime()) {
			modTime := stat.ModTime()
			newest = &modTime
			path = &currPath
		}
	}
	return path, newest, nil
}

// FindNewestText
// Given a directory, recursively scans and returns the path and modified time
// for the newest `.txt` file.
func FindNewestText(dirPath string) (path *string, newest *time.Time,
	err error) {
	matches, err := GlobTexts(dirPath)
	if err != nil {
		return nil, nil, err
	}
	return FindNewestPath(&matches)
}

// FindNewestDir
// Given a directory, recursively scans and returns the path and modified time
// for the directory that contains the most recent `.txt` modification.
func FindNewestDir(dirPath string) (path *string, newest *time.Time,
	err error) {
	fileMatches, err := GlobTexts(dirPath)
	if err != nil {
		return nil, nil, err
	}
	// Find all directories, as removed files will modify the time of the
	// containing directory.
	directories := make(map[string]bool, 0)
	for matchIdx := range fileMatches {
		currPath := fileMatches[matchIdx]
		directories[filepath.Dir(currPath)] = true
	}
	directoryMatches := make([]string, 0)
	for dir, _ := range directories {
		directoryMatches = append(directoryMatches, dir)
	}
	return FindNewestPath(&directoryMatches)
}

// ReadTexts
// Consumes a directory path and recursively scans for `.txt` files, producing
// a TextsIterator function that yields the text file contents.
func ReadTexts(dirPath string, sanitize bool) (TextsIterator, error) {
	matches, err := GlobTexts(dirPath)
	if err != nil {
		return nil, err
	}
	numMatches := len(matches)
	texts := make(chan *string, 2)

	go func() {
		for matchIdx := 0; matchIdx < numMatches; matchIdx++ {
			path := matches[matchIdx]
			if textBytes, readErr := os.ReadFile(path); readErr != nil {
				close(texts)
				log.Fatal(readErr)
			} else {
				log.Print("Reading ", path)
				text := string(textBytes)
				if sanitize {
					text = SanitizeText(text)
				}
				texts <- &text
			}
		}
		close(texts)
	}()

	return func() *string {
		if text, more := <-texts; !more {
			return nil
		} else {
			return text
		}
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
	s = strings.ReplaceAll(s, "\\n", "\n")
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
func (tt *TextsTokenizer) InitTokenizer() (*gpt_bpe.GPTEncoder, error) {
	tokenizerPtr, ok := tokenizers[tt.TokenizerId]
	if !ok {
		var tokErr error
		tokenizerPtr, tokErr = gpt_bpe.NewEncoder(tt.TokenizerId)
		if tokErr != nil {
			return nil, tokErr
		} else {
			tokenizers[tt.TokenizerId] = tokenizerPtr
			return tokenizerPtr, nil
		}
	}
	return tokenizerPtr, nil
}

func (tt TextsTokenizer) TokenizeTexts(
	nextText TextsIterator) (ContextsIterator, error) {
	tokenizerPtr, tokErr := tt.InitTokenizer()
	if tokErr != nil {
		return nil, tokErr
	}
	tokenizer := *tokenizerPtr
	var padToken, endOfText gpt_bpe.Token
	if tt.PadToken == "" {
		padToken = tokenizer.PadToken
	} else {
		var padErr error
		padToken, padErr = getAndCheckToken(&tokenizer, tt.PadToken,
			"PadToken")
		if padErr != nil {
			return nil, padErr
		}
	}
	if tt.EndOfText == "" {
		endOfText = tokenizer.EosToken
	} else {
		var eotErr error
		endOfText, eotErr = getAndCheckToken(&tokenizer, tt.EndOfText,
			"EndOfText")
		if eotErr != nil {
			return nil, eotErr
		}
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

	var tokens gpt_bpe.Tokens
	var text *string
	var done bool
	var numTokens, idx, begin, boundaryIdx int

	// Consume texts from `nextText()` and tokenize as a `goroutine`.
	tokenizedTexts := make(chan gpt_bpe.Tokens, 2)
	nextTokenized := func() {
		for {
			text = nextText()
			if text != nil {
				tokenized := tokenizer.Encode(text)
				tokenizedTexts <- *tokenized
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
			numTokens = len(tokens)
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
				token := (tokens)[idx]
				// Mark the position of the last `boundary` token we've seen.
				if token == boundary {
					boundaryIdx = idx
				}
				// Determine if we're at least `contextSize` yet, and if so
				// we do the finalization of this context.
				if idx-begin+len(prior) >= contextSize {
					var chunk gpt_bpe.Tokens
					// If we have `prior` from a prior text, we prepend the
					// beginning of this text.
					if len(prior) > 0 {
						chunk = append(prior, (tokens)[begin:]...)
						prior = make(gpt_bpe.Tokens, 0)
					} else {
						chunk = (tokens)[begin:]
					}

					if doUnitrim {
						var endAt int
						chunk, endAt = tokenizer.AlignAndSizeTokens(&chunk,
							contextSize)
						idx = begin + endAt
					} else if len(chunk) > contextSize {
						chunk = (tokens)[:contextSize]
					} else {
						idx = begin + len(chunk)
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
				prior = append(prior, (tokens)[begin:]...)
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
func WriteContexts(outPath string, nextContext ContextsIterator,
	encoder *gpt_bpe.GPTEncoder) (int, error) {
	totalTokens := 0
	outFile, err := os.OpenFile(outPath, os.O_TRUNC|os.O_WRONLY|os.O_CREATE,
		755)
	if err != nil {
		return 0, err
	}
	contexts := make(chan gpt_bpe.Tokens, 2)

	go func() {
		for {
			context := nextContext()
			if context == nil {
				close(contexts)
				break
			} else {
				contexts <- *context
				if encoder != nil {
					println(len(*context))
					println("=========================================")
					println(encoder.Decode(context))
				}
			}
		}
	}()

	for {
		context, more := <-contexts
		if !more {
			break
		}
		binContext := context.ToBin()
		if _, writeErr := outFile.Write(*binContext); writeErr != nil {
			return totalTokens, writeErr
		}
		totalTokens += len(context)
	}
	return totalTokens, nil
}

func init() {
	tokenizers = make(map[string]*gpt_bpe.GPTEncoder, 0)
	tokenizers["gpt2"] = &gpt_bpe.GPT2Encoder
	tokenizers["pile"] = &gpt_bpe.PileEncoder
}

func main() {
	tokenizerId := flag.String("tokenizer", "gpt2",
		"tokenizer to use [gpt2, pile, huggingface-id]")
	contextSize := flag.Int("context", 2048, "context size")
	showContexts := flag.Bool("show_contexts", false,
		"show contexts as they are tokenized")
	endOfText := flag.String("eot", "",
		"end of text token to split texts, can be token or int16 "+
			"token_id")
	padToken := flag.String("pad", "",
		"pad token to pad out contexts, can be <|padding|>, or an "+
			"int16 token_id")
	boundaryToken := flag.String("boundary", "\n",
		"boundary token to split contexts on, can be a string token "+
			"or int16 token_id")
	outputFile := flag.String("output", "tokenized.chunk",
		"tokenized output file")
	inputDir := flag.String("input", "",
		"input directory")
	unitrimBool := flag.Bool("no_unitrim", false,
		"do not trim contexts to valid unicode")
	forceRetokenization := flag.Bool("retokenize", false,
		"force retokenization even if tokenizer output is newer")
	sanitizeBool := flag.Bool("sanitize", false,
		"sanitize inputs of whitespace issues")
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
	textsTokenizer.Unitrim = !*unitrimBool

	if !*forceRetokenization {
		if outStat, outErr := os.Stat(*outputFile); !errors.Is(outErr,
			os.ErrNotExist) && outErr != nil {
			log.Fatal(outErr)
		} else if errors.Is(outErr, os.ErrNotExist) {
			log.Printf("Creating %s", *outputFile)
		} else if newestPath, newestModTime, newestErr := FindNewestText(
			*inputDir); newestErr != nil {
			log.Fatal(newestErr)
		} else if newestModTime != nil && newestModTime.Before(
			outStat.ModTime()) {
			log.Printf("Newest source `%s` is older than `%s`, "+
				"not retokenizing. "+
				"Use -retokenize to force retokenization.", *newestPath,
				*outputFile)
			os.Exit(0)
		} else if newestDir, newestDirModTime, newestDirErr := FindNewestDir(
			*inputDir); newestDirErr != nil {
			log.Fatal(newestDirErr)
		} else if newestDirModTime != nil && newestDirModTime.Before(
			outStat.ModTime()) {
			log.Printf("Data source directory `%s` has no changes since `%s"+
				"was tokenized. Use -retokenize to force retokenization.",
				*newestDir, *outputFile)
		}
	}
	if _, tokErr := textsTokenizer.InitTokenizer(); tokErr != nil {
		log.Fatal(tokErr)
	}

	if nextText, err := ReadTexts(*inputDir, *sanitizeBool); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts, tokErr := textsTokenizer.TokenizeTexts(
			nextText)
		if tokErr != nil {
			log.Fatal(tokErr)
		}
		var enc *gpt_bpe.GPTEncoder
		if *showContexts {
			enc, _ = gpt_bpe.NewEncoder(*tokenizerId)
		}
		total, writeErr := WriteContexts(*outputFile, contexts, enc)
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		log.Printf("%d tokens in %0.2fs", total,
			time.Now().Sub(begin).Seconds())
	}
}

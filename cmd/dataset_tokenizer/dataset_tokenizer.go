package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"github.com/wbrown/gpt_bpe"
	"github.com/yargevad/filepathx"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

var tokenizers map[string]*gpt_bpe.GPTEncoder

type TextsIterator func() io.RuneReader

// GlobTexts
// Given a directory path, recursively finds all `.txt` files, returning a
// slice of paths.
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

// FindNewestPath
// Given a directory path, recursively scans and returns the path and modified
// time for the newest `.txt` file.
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
	for dir := range directories {
		directoryMatches = append(directoryMatches, dir)
	}
	return FindNewestPath(&directoryMatches)
}

// ReadTexts
// Consumes a directory path and recursively scans for `.txt` files, producing
// a TextsIterator function that yields the text file as an io.Reader type.
func ReadTexts(dirPath string, sanitize bool) (TextsIterator, error) {
	matches, err := GlobTexts(dirPath)
	if err != nil {
		return nil, err
	}
	numMatches := len(matches)

	type namedRuneReader struct {
		path   string
		reader io.RuneReader
	}

	// We pre-emptively do the work to set up the buffers for the next file,
	// while the pror file is being consumed.
	runeReaders := make(chan namedRuneReader, 1)
	go func() {
		for matchIdx := 0; matchIdx < numMatches; matchIdx++ {
			path := matches[matchIdx]
			if fileReader, openErr := os.Open(path); openErr != nil {
				log.Fatal(openErr)
			} else {
				if sanitize {
					runeReaders <- namedRuneReader{
						path,
						CreateTextSanitizer(fileReader)}
				} else {
					runeReaders <- namedRuneReader{
						path,
						bufio.NewReader(fileReader)}
				}
			}
		}
		close(runeReaders)
	}()

	return func() io.RuneReader {
		if reader, ok := <-runeReaders; !ok {
			return nil
		} else {
			log.Print("Reading ", reader.path)
			return reader.reader
		}
	}, nil
}

// TextsTokenizer
// A struct that encapsulates the configuration for a streaming tokenizer.
type TextsTokenizer struct {
	TokenizerId string
	ContextSize int
	Boundary    string
	PadToken    string
	EndOfText   string
	Unitrim     bool
}

// NewTextsTokenizer
// Creates a new TextsTokenizer struct with the default configuration.
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

// getAndCheckToken
// Check if s is a valid token via tokenizer table lookup; if not, we try
// encoding and checking if it is a valid single token returned.
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

// TokenizeTexts
// Consumes a TextsIterator and produces a ContextsIterator iterator function
// that returns tokenized contexts that are fixed and padded out to
// `contextSize`.
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

	var tokens gpt_bpe.Tokens
	var done bool
	var numTokens, idx, begin, boundaryIdx int

	// Consume texts from `nextText()` and tokenize as a `goroutine`.
	tokenizedTexts := make(chan gpt_bpe.Tokens, 4)
	nextTokenized := func() {
		for {
			runeReader := nextText()
			if runeReader != nil {
				encodeChunk := tokenizer.StreamingEncode(runeReader)
				for {
					tokenized := encodeChunk(contextSize * 8)
					if tokenized == nil {
						tokenizedTexts <- gpt_bpe.Tokens{endOfText}
						break
					}
					tokenizedTexts <- *tokenized
				}
			} else {
				close(tokenizedTexts)
				break
			}
		}
	}
	go nextTokenized()

	// Consumes tokenized texts and resets closured states for token blocks.
	moreTokens := func() {
		moreTokens, more := <-tokenizedTexts
		tokens = append(tokens, moreTokens...)
		numTokens = len(tokens)
		if more {
			done = false
		} else {
			done = true
		}
	}

	// Prime the pump by initializing the states.
	moreTokens()

	// Return an iterator function that returns token chunks that are always
	// `contextSize` tokens.
	nextContext := func() *gpt_bpe.Tokens {
		if len(tokens)-idx < contextSize*4 {
			moreTokens()
		}
		// Loop until we get a full token chunk.
		for {
			if numTokens == 0 {
				return nil
			} else if done && idx == numTokens {
				// We're completely done and have no more token chunks to
				// return, so we flush out and pad the last chunk.
				chunk := tokens[begin:]
				padSize := contextSize - len(chunk)
				if padSize > 0 {
					for padIdx := 0; padIdx < padSize; padIdx += 1 {
						chunk = append(chunk, padToken)
					}
				}
				tokens = tokens[:0]
				idx = 0
				numTokens = 0
				begin = 0
				return &chunk
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
				currWindow := idx - begin
				if currWindow >= contextSize {
					chunk := (tokens)[begin:]

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
					if idx > contextSize*6 {
						tokens = tokens[idx:]
						begin = 0
						idx = 0
					} else {
						begin = idx
					}
					numTokens = len(tokens)
					return &chunk
				}
				idx += 1
				if len(tokens)-idx < contextSize*2 {
					moreTokens()
				}
			}
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
		// *showContexts = true
		if *showContexts {
			enc, _ = gpt_bpe.NewEncoder(*tokenizerId)
		}
		total, writeErr := WriteContexts(*outputFile, contexts, enc)
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", total,
			duration, float64(total)/duration)

	}
}

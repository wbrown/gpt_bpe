package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/wbrown/gpt_bpe"
	"github.com/yargevad/filepathx"
)

var tokenizers map[string]func() gpt_bpe.GPTEncoder

type TextsIterator func() (string, io.RuneReader)

type PathInfo struct {
	Path    string
	Size    int64
	ModTime time.Time
	Dir     bool
}

// GlobTexts
// Given a directory path, recursively finds all `.txt` and `.jsonl` files,
// returning a slice of PathInfo.
func GlobTexts(dirPath string) (pathInfos []PathInfo, err error) {
	textPaths, err := filepathx.Glob(dirPath + "/**/*.txt")
	if err != nil {
		return nil, err
	}
	jsonlPaths, err := filepathx.Glob(dirPath + "/**/*.jsonl")
	if err != nil {
		return nil, err
	}
	filePaths := append(textPaths, jsonlPaths...)

	numMatches := len(filePaths)
	if numMatches == 0 {
		return nil, errors.New(fmt.Sprintf(
			"%s does not contain any .txt or .jsonl files", dirPath))
	}
	pathInfos = make([]PathInfo, numMatches)
	for matchIdx := range filePaths {
		currPath := filePaths[matchIdx]
		if stat, statErr := os.Stat(currPath); statErr != nil {
			return nil, statErr
		} else {
			pathInfos[matchIdx] = PathInfo{
				Path:    currPath,
				Size:    stat.Size(),
				ModTime: stat.ModTime(),
				Dir:     stat.IsDir(),
			}
		}
	}
	return pathInfos, nil
}

func SortPathInfoBySize(pathInfos []PathInfo, ascending bool) {
	if ascending {
		sort.Slice(pathInfos, func(i, j int) bool {
			return pathInfos[i].Size < pathInfos[j].Size
		})
	} else {
		sort.Slice(pathInfos, func(i, j int) bool {
			return pathInfos[i].Size > pathInfos[j].Size
		})
	}
}

func SortPathInfoByPath(pathInfos []PathInfo, ascending bool) {
	if ascending {
		sort.Slice(pathInfos, func(i, j int) bool {
			return pathInfos[i].Path < pathInfos[j].Path
		})
	} else {
		sort.Slice(pathInfos, func(i, j int) bool {
			return pathInfos[i].Path > pathInfos[j].Path
		})
	}
}

func GetFileInfo(paths []string) ([]os.FileInfo, error) {
	fileInfos := make([]os.FileInfo, 0)
	for _, path := range paths {
		if fileInfo, err := os.Stat(path); err != nil {
			return nil, err
		} else {
			fileInfos = append(fileInfos, fileInfo)
		}
	}
	return fileInfos, nil
}

func GetPathsFromFileInfo(fileInfos []os.FileInfo) (paths []string) {
	for _, fileInfo := range fileInfos {
		paths = append(paths, fileInfo.Name())
	}
	return paths
}

func ShufflePathInfos(pathInfos []PathInfo) {
	for i := len(pathInfos) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		pathInfos[i], pathInfos[j] = pathInfos[j], pathInfos[i]
	}
}

// FindNewestPath
// Given a directory path, recursively scans and returns the path and modified
// time for the newest `.txt` file.
func FindNewestPath(paths []PathInfo) (path *string, newest *time.Time,
	err error) {
	var newestPath string
	var newestTime *time.Time
	for _, pathInfo := range paths {
		if newestTime == nil || newestTime.Before(pathInfo.ModTime) {
			newestTime = &pathInfo.ModTime
			newestPath = pathInfo.Path
		}
	}
	return &newestPath, newestTime, nil
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
	return FindNewestPath(matches)
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
	directories := make([]PathInfo, 0)
	for matchIdx := range fileMatches {
		if fileMatches[matchIdx].Dir {
			directories = append(directories, fileMatches[matchIdx])
		}
	}
	return FindNewestPath(directories)
}

type namedRuneReader struct {
	path   string
	reader io.RuneReader
}

// ReadTexts
// Consumes a directory path and recursively scans for `.txt` files, producing
// a TextsIterator function that yields the text file as an io.Reader type.
func ReadTexts(dirPath string,
	sanitize bool,
	sortSpec string,
	numReaderThreads int,
) (chan namedRuneReader, error) {
	matches, err := GlobTexts(dirPath)
	if err != nil {
		return nil, err
	}

	if sortSpec != "" && sortSpec != "none" && sortSpec != "shuffle" {
		if sortSpec == "size_ascending" {
			SortPathInfoBySize(matches, true)
		} else if sortSpec == "size_descending" {
			SortPathInfoBySize(matches, false)
		} else if sortSpec == "path_ascending" {
			SortPathInfoByPath(matches, true)
		} else if sortSpec == "path_descending" {
			SortPathInfoByPath(matches, false)
		} else if sortSpec == "random" {
			ShufflePathInfos(matches)
		} else {
			return nil, errors.New(fmt.Sprintf(
				"Invalid sort spec: %s", sortSpec))
		}
	}

	// We pre-emptively do the work to set up the buffers for the next files,
	// while the prior file is being consumed.
	runeReaders := make(chan namedRuneReader, 64)
	paths := make(chan PathInfo, 64)
	wg := sync.WaitGroup{}
	startReader := func() {
		for {
			if path, ok := <-paths; ok {
				if fileReader, openErr := os.Open(path.Path); openErr != nil {
					log.Fatal(openErr)
				} else {
					if strings.HasSuffix(path.Path, ".jsonl") {
						// Split JSONL files into individual JSON objects.
						jsonlReader := bufio.NewReader(fileReader)
						idx := 0
						for {
							jsonObject, rErr := jsonlReader.ReadBytes('\n')
							if rErr != nil {
								if rErr == io.EOF {
									break
								} else {
									log.Fatal(rErr)
								}
							}
							// Decode the JSON object.
							var jsonObjectMap map[string]interface{}
							if jErr := json.Unmarshal(jsonObject,
								&jsonObjectMap); jErr != nil {
								log.Printf(
									"JSONL object %d in %s is not valid JSON: %s",
									idx, path.Path, jErr)
								continue
							}
							// Extract the text field.
							text, ok := jsonObjectMap["text"]
							if !ok {
								log.Fatal("JSONL object missing text field")
							}
							textString, ok := text.(string)
							if !ok {
								log.Fatal("JSONL object text field not string")
							}
							subPath := fmt.Sprintf("%s[%d]", path.Path, idx)
							// Create our rune reader.
							if sanitize {
								runeReaders <- namedRuneReader{
									subPath,
									CreateTextSanitizer(
										strings.NewReader(textString))}
							} else {
								runeReaders <- namedRuneReader{
									subPath,
									strings.NewReader(textString)}
							}
							idx++
						}
					} else {
						if sanitize {
							runeReaders <- namedRuneReader{
								path.Path,
								CreateTextSanitizer(fileReader)}
						} else {
							runeReaders <- namedRuneReader{
								path.Path,
								bufio.NewReaderSize(fileReader,
									8*1024*1024)}
						}
					}
				}
			} else {
				break
			}
		}
		wg.Done()
	}
	for readerIdx := 0; readerIdx < numReaderThreads; readerIdx++ {
		wg.Add(1)
		go startReader()
	}
	go func() {
		for matchIdx := range matches {
			paths <- matches[matchIdx]
		}
		close(paths)
		wg.Wait()
		close(runeReaders)
	}()
	return runeReaders, nil
}

// TextsTokenizer
// A struct that encapsulates the configuration for a streaming tokenizer.
type TextsTokenizer struct {
	TokenizerId     string
	ContextSize     int
	Boundary        string
	BoundaryBegin   bool
	BoundaryOverlap int
	PadToken        string
	EndOfText       string
	Unitrim         bool
}

// NewTextsTokenizer
// Creates a new TextsTokenizer struct with the default configuration.
func NewTextsTokenizer() TextsTokenizer {
	return TextsTokenizer{
		"gpt2",
		2048,
		"\n",
		false,
		-1,
		"<|endoftext|>",
		"<|padding|>",
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
	var encoderPtr *gpt_bpe.GPTEncoder
	tokenizerFn, ok := tokenizers[tt.TokenizerId]
	if !ok {
		var tokErr error
		encoderPtr, tokErr = gpt_bpe.NewEncoder(tt.TokenizerId)
		if tokErr != nil {
			return nil, tokErr
		}
	} else {
		encoder := tokenizerFn()
		encoderPtr = &encoder
	}
	return encoderPtr, nil
}

func (tt *TextsTokenizer) PartitionBoundary(
	boundaryIdxes *[]int,
	begin int,
	idx int,
) int {
	if tt.Boundary == "" && tt.BoundaryOverlap >= 0 {
		idx = begin + tt.BoundaryOverlap
	} else if len(*boundaryIdxes) > 0 {
		var boundaryIdx int
		if tt.BoundaryOverlap == -1 {
			boundaryIdx = (*boundaryIdxes)[len(*boundaryIdxes)-1]
		} else {
			// Find the closest boundary token to the index
			// of the BoundaryOverlap.
			smaller := 0
			larger := 0
			for _, boundaryIdx = range *boundaryIdxes {
				if boundaryIdx < begin+tt.BoundaryOverlap {
					smaller = boundaryIdx
				} else {
					larger = boundaryIdx
					break
				}
			}
			if smaller == 0 {
				boundaryIdx = larger
			} else if larger == 0 {
				boundaryIdx = smaller
			} else {
				if tt.BoundaryOverlap-smaller <
					larger-tt.BoundaryOverlap {
					boundaryIdx = smaller
				} else {
					boundaryIdx = larger
				}
			}
		}

		if boundaryIdx > 0 && boundaryIdx <= begin+tt.ContextSize {
			boundaryOffset := 0
			if !tt.BoundaryBegin {
				boundaryOffset = 1
			}
			if idx-boundaryIdx+boundaryOffset <= tt.ContextSize {
				idx = boundaryIdx + boundaryOffset
			}
		}
	}
	return idx
}

func (tt TextsTokenizer) TokenizeTexts(
	texts chan namedRuneReader,
	indexPath string,
) (chan gpt_bpe.Tokens, error) {
	tokenizerPtr, tokErr := tt.InitTokenizer()
	if tokErr != nil {
		return nil, tokErr
	}
	tokenizer := *tokenizerPtr
	var endOfText gpt_bpe.Token
	if tt.EndOfText == "" {
		endOfText = tokenizer.EosToken
	} else {
		var eotErr error
		if endOfText, eotErr = getAndCheckToken(&tokenizer,
			tt.EndOfText, "EndOfText"); eotErr != nil {
			return nil, eotErr
		}
	}

	tokenizedTexts := make(chan gpt_bpe.Tokens, 32)

	// Our index handle.
	indexFile, iErr := os.Create(indexPath)
	if iErr != nil {
		return nil, iErr
	}

	currOffset := 0
	nextTokenized := func() {
		for {
			waitBegin := time.Now()
			runeReader, more := <-texts
			if more {
				waitDuration := time.Since(waitBegin)
				beginTs := time.Now()
				tokenCt := 0
				encodeChunk := tokenizer.StreamingEncode(runeReader.reader)
				for {
					tokenized := encodeChunk(16384)
					if tokenized == nil {
						tokenizedTexts <- gpt_bpe.Tokens{endOfText}
						tokenCt += 1
						break
					} else {
						tokenCt += len(*tokenized)
					}
					tokenizedTexts <- *tokenized
				}
				duration := time.Since(beginTs)
				// If we took longer than a millisecond, round to the nearest
				// millisecond.
				var roundDuration time.Duration
				if duration.Round(time.Millisecond) > 0 {
					roundDuration = duration.Round(time.Millisecond)
				} else {
					roundDuration = duration
				}
				log.Printf("%s tokenized in %s (%d tokens/s, %s wait)",
					runeReader.path, roundDuration,
					int(float64(tokenCt)/duration.Seconds()),
					waitDuration)
				indexFile.WriteString(fmt.Sprintf(
					"{\"path\": \"%s\", \"offset\": %d, \"tokens\": %d}\n",
					runeReader.path, currOffset, tokenCt))
				currOffset += tokenCt
			} else {
				close(tokenizedTexts)
				indexFile.Close()
				break
			}
		}
	}
	go nextTokenized()
	return tokenizedTexts, nil
}

// TokenizeTextsToContexts
// Consumes a TextsIterator and produces a ContextsIterator iterator function
// that returns tokenized contexts that are fixed and padded out to
// `contextSize`.
func (tt TextsTokenizer) TokenizeTextsToContexts(
	texts chan namedRuneReader,
) (chan gpt_bpe.Tokens, error) {
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
	var numTokens, idx, begin int
	boundaryIdxes := make([]int, 0)

	// Consume texts from `nextText()` and tokenize as a `goroutine`.
	tokenizedTexts := make(chan gpt_bpe.Tokens, 16)
	nextTokenized := func() {
		for {
			runeReader, more := <-texts
			if more {
				encodeChunk := tokenizer.StreamingEncode(runeReader.reader)
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
				// If this is a 'boundary' token, add it to our list.
				if token == boundary {
					boundaryIdxes = append(boundaryIdxes, idx)
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
					// We had one or more boundary tokens in our last context,
					// so depending on the BoundaryOverlap, use the last
					// boundary or the boundary closest to the BounderOverlap
					// index. This effectively copies the chunk from that point
					// on into the next returned context.
					idx = tt.PartitionBoundary(&boundaryIdxes, begin, idx)

					// We were given a hard index to use as the chunk boundary,
					// and it may not be a complete unicode character, so we
					// need to align it to a valid unicode character.
					if boundary == 65535 && doUnitrim {
						// Ensure that our next chunk is aligned to valid
						// unicode.
						_, offset := tokenizer.AlignAndSizeTokens(&chunk,
							idx-begin)
						idx = begin + offset
						if offset != tt.BoundaryOverlap {
							println("idx: ", idx, " offset:", offset)
						}
					}

					boundaryIdxes = boundaryIdxes[:0]

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

	go func() {
		for {
			context := nextContext()
			if context == nil {
				close(tokenizedTexts)
				break
			} else {
				tokenizedTexts <- *context
			}
		}
	}()

	return tokenizedTexts, nil
}

// WriteContexts
// Consumes a ContextsIterator function and serializes the contexts to an
// aligned binary file.
func WriteContexts(outPath string, contexts chan gpt_bpe.Tokens,
	encoder *gpt_bpe.GPTEncoder, sampling int, shuffle bool) (int, error) {
	totalTokens := 0
	outFile, err := os.OpenFile(outPath, os.O_TRUNC|os.O_RDWR|os.O_CREATE,
		0755)
	if err != nil {
		return 0, err
	}
	sampledContexts := make(chan gpt_bpe.Tokens, 16)

	go func() {
		samplingIdx := 0
		for {
			context, ok := <-contexts
			if !ok {
				close(sampledContexts)
				break
			} else {
				// Ignore every `sampling` percent context (rounded to int)
				if sampling == 100 || (samplingIdx%20) < int(sampling/5) {
					sampledContexts <- context
					if encoder != nil {
						println(len(context))
						println("======================================")
						println(encoder.Decode(&context))
					}
				}
				samplingIdx += 1
			}
		}
	}()

	endpos := 0
	var buf []byte
	var contextSize int
	var target int64

	// Sometimes it is requested that we shuffle all contexts as they are
	// written
	for {
		context, more := <-sampledContexts
		if !more {
			break
		}
		binContext := context.ToBin()
		// We keep track of the final file position
		if endpos == 0 {
			// On the first context, we discern the context size and make the
			// appropriately sized buffer
			contextSize = len(*binContext)
			buf = make([]byte, contextSize)

		}

		// We select a random position in the buffer that is a multiple of the
		// context size
		if endpos == 0 {
			target = 0
		} else {
			target = int64(rand.Intn((endpos)/contextSize)) * int64(contextSize)
		}

		// If shuffling, we store the context found at the target position in
		// the buffer and write it to the end of the file, writing the new
		// context to the target position
		if endpos != 0 && shuffle {
			if _, err := outFile.ReadAt(buf, target); err != nil {
				return totalTokens, err
			}
		} else if shuffle {
			//write the buffer to the end of the file
			if _, err := outFile.Write(*binContext); err != nil {
				return totalTokens, err
			}
		}

		if endpos > 0 && shuffle {
			// Overwrite binContext to the location of the context we just read
			if _, err := outFile.WriteAt(*binContext, target); err != nil {
				return totalTokens, err
			}

			// Write the context we just read and replaced to the end of the
			// file
			if _, err := outFile.Write(buf); err != nil {
				return totalTokens, err
			}
		} else if !shuffle {
			// Else, we just write the context to the end of the file as usual
			if _, err := outFile.Write(*binContext); err != nil {
				return totalTokens, err
			}
		}

		totalTokens += len(context)
		endpos += len(*binContext)
	}

	return totalTokens, nil
}

func init() {
	tokenizers = make(map[string]func() gpt_bpe.GPTEncoder, 0)
	tokenizers["gpt2"] = gpt_bpe.NewGPT2Encoder
	tokenizers["pile"] = gpt_bpe.NewPileEncoder
	tokenizers["nerdstash"] = gpt_bpe.NewNerdstashEncoder
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
	boundaryBegin := flag.Bool("boundary_begin", false,
		"whether to treat the boundary token as a beginning token for"+
			"a context")
	boundaryOverlap := flag.Int("boundary_overlap", -1,
		"token index in context to approximately overlap contexts on, "+
			"-1 for last boundary token in context")
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
	reorderPaths := flag.String("reorder", "",
		"reorder input files to specification [size_ascending, "+
			"size_descending, name_ascending, name_descending, random, shuffle, none]")
	sampling_str := flag.String("sampling", "100", "a integer value from 0-100 "+
		"which tells the tokenizer how many chunks to discard in %, 60 keeps 60%% chunks")
	streaming_encode := flag.Bool("streaming_encode", false,
		"use streaming encode, which writes to disk as it encodes, "+
			"rather than buffering into contexts")
	numThreads := flag.Int("threads", 4,
		"number of tokenization threads to use")
	numReaderThreads := flag.Int("reader_threads", 2,
		"number of I/O reader threads to use")
	flag.Parse()
	if *inputDir == "" {
		flag.Usage()
		log.Fatal("Must provide -input for directory source")
	}
	sampling, err := strconv.Atoi(*sampling_str)
	if err != nil {
		log.Fatal("Sampling parameter must be an integer")
	}

	if sampling > 100 || sampling < 0 {
		log.Fatal("Sampling parameter out of the 0-100 bounds")
	}

	log.Printf("Tokenizer definition: %s\n", *tokenizerId)
	log.Printf("Tokenizer input source: %s\n", *inputDir)
	log.Printf("Tokenizer output: %s\n", *outputFile)
	log.Printf("Tokenizer reordering method: %s\n", *reorderPaths)
	log.Printf("Sampling amount (in %s tokens kept): %d%s\n",
		"%", sampling, "%")

	if *reorderPaths != "" {
		if *reorderPaths != "size_ascending" &&
			*reorderPaths != "size_descending" &&
			*reorderPaths != "name_ascending" &&
			*reorderPaths != "name_descending" &&
			*reorderPaths != "random" &&
			*reorderPaths != "shuffle" &&
			*reorderPaths != "none" {
			log.Fatal("Invalid reorder specification")
		}
	}

	textsTokenizer := NewTextsTokenizer()
	textsTokenizer.ContextSize = *contextSize
	textsTokenizer.TokenizerId = *tokenizerId
	textsTokenizer.EndOfText = *endOfText
	textsTokenizer.PadToken = *padToken
	textsTokenizer.Boundary = *boundaryToken
	textsTokenizer.BoundaryBegin = *boundaryBegin
	textsTokenizer.BoundaryOverlap = *boundaryOverlap
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

	if textReaders, err := ReadTexts(*inputDir, *sanitizeBool,
		*reorderPaths, *numReaderThreads); err != nil {
		log.Fatal(err)
	} else {
		numTokens := 0
		begin := time.Now()
		if *streaming_encode {
			wg := sync.WaitGroup{}
			for threadIdx := 0; threadIdx < *numThreads; threadIdx++ {
				wg.Add(1)
				go func(threadId int) {
					var contexts chan gpt_bpe.Tokens
					var tokErr error
					indexFilePath := fmt.Sprintf("%s.%d.index",
						*outputFile, threadId)
					outputFilePath := fmt.Sprintf("%s.%d.tokens",
						*outputFile, threadId)
					contexts, tokErr = textsTokenizer.TokenizeTexts(textReaders,
						indexFilePath)
					if tokErr != nil {
						log.Fatal(tokErr)
					}
					total, writeErr := WriteContexts(outputFilePath, contexts,
						nil, sampling, false)
					if writeErr != nil {
						log.Fatal(writeErr)
					}
					numTokens += total
					wg.Done()
				}(threadIdx)
			}
			wg.Wait()
		} else {
			var contexts chan gpt_bpe.Tokens
			var tokErr error
			contexts, tokErr = textsTokenizer.TokenizeTextsToContexts(
				textReaders)
			}
			if tokErr != nil {
				log.Fatal(tokErr)
			}
			var enc *gpt_bpe.GPTEncoder
			if *showContexts {
				enc, _ = gpt_bpe.NewEncoder(*tokenizerId)
			}
			var writeErr error
			numTokens, writeErr = WriteContexts(*outputFile, contexts, enc,
				sampling,
				*reorderPaths == "shuffle")
			if writeErr != nil {
				log.Fatal(writeErr)
			}
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", numTokens,
			duration, float64(numTokens)/duration)
	}
}

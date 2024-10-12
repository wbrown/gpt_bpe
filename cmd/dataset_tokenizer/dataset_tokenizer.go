package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/wbrown/gpt_bpe"
	"github.com/wbrown/gpt_bpe/resources"
	"github.com/yargevad/filepathx"
)

type PathInfo struct {
	Path    string
	Size    int64
	ModTime time.Time
	Dir     bool
}

type S3Client interface {
	GetObject(input *s3.GetObjectInput) (*s3.GetObjectOutput, error)
	ListObjectsV2(input *s3.ListObjectsV2Input) (
		*s3.ListObjectsV2Output,
		error,
	)
}

// GlobTexts
// Given a directory path, recursively finds all `.txt` and `.jsonl` files,
// returning a slice of PathInfo.
func GlobTexts(dirPath string) (pathInfos []PathInfo, err error) {
	// If the path is a file, return it.
	if stat, _ := os.Stat(dirPath); stat != nil && !stat.IsDir() {
		return []PathInfo{{
			Path:    dirPath,
			Size:    stat.Size(),
			ModTime: stat.ModTime(),
			Dir:     stat.IsDir(),
		}}, nil
	}

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
		return nil, fmt.Errorf(
			"%s does not contain any .txt or .jsonl files", dirPath,
		)
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
		sort.Slice(
			pathInfos, func(i, j int) bool {
				return pathInfos[i].Size < pathInfos[j].Size
			},
		)
	} else {
		sort.Slice(
			pathInfos, func(i, j int) bool {
				return pathInfos[i].Size > pathInfos[j].Size
			},
		)
	}
}

func SortPathInfoByPath(pathInfos []PathInfo, ascending bool) {
	if ascending {
		sort.Slice(
			pathInfos, func(i, j int) bool {
				return pathInfos[i].Path < pathInfos[j].Path
			},
		)
	} else {
		sort.Slice(
			pathInfos, func(i, j int) bool {
				return pathInfos[i].Path > pathInfos[j].Path
			},
		)
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
func FindNewestPath(paths []PathInfo) (
	path *string, newest *time.Time,
	err error,
) {
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
func FindNewestText(dirPath string) (
	path *string, newest *time.Time,
	err error,
) {
	matches, err := GlobTexts(dirPath)
	if err != nil {
		return nil, nil, err
	}
	return FindNewestPath(matches)
}

// FindNewestDir
// Given a directory, recursively scans and returns the path and modified time
// for the directory that contains the most recent `.txt` modification.
func FindNewestDir(dirPath string) (
	path *string, newest *time.Time,
	err error,
) {
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

func resolveSortSpec(matches []PathInfo, sortSpec string) (err error) {
	if sortSpec == "" || sortSpec == "none" || sortSpec == "shuffle" {
		ShufflePathInfos(matches)
	} else if sortSpec == "size_ascending" {
		SortPathInfoBySize(matches, true)
	} else if sortSpec == "size_descending" {
		SortPathInfoBySize(matches, false)
	} else if sortSpec == "path_ascending" {
		SortPathInfoByPath(matches, true)
	} else if sortSpec == "path_descending" {
		SortPathInfoByPath(matches, false)
	} else {
		return fmt.Errorf("invalid sort spec: %s", sortSpec)
	}
	return nil
}

// getObjectsS3Recursively retrieves objects recursively from an S3 bucket and sends them to the objects channel.
func getObjectsS3Recursively(
	svc S3Client,
	bucketName, prefix string,
	objects chan<- *s3.Object,
) {
	var continuationToken *string
	for {
		params := &s3.ListObjectsV2Input{
			Bucket:            aws.String(bucketName),
			Prefix:            aws.String(prefix),
			ContinuationToken: continuationToken,
		}

		resp, err := svc.ListObjectsV2(params)
		if err != nil {
			log.Printf("Error listing objects: %v", err)
			return
		}

		for _, obj := range resp.Contents {
			key := *obj.Key
			if strings.HasSuffix(key, ".txt") || strings.HasSuffix(
				key, ".jsonl",
			) {
				objects <- obj
			}
		}

		if !*resp.IsTruncated {
			break
		}

		continuationToken = resp.NextContinuationToken
	}
}

// fetchJSONLFileS3 reads a JSONL file from S3, extracts the "text" key, and return it as a string with spaces.
func fetchJSONLFileS3(svc S3Client, bucketName, objectKey string) (
	string,
	error,
) {
	params := &s3.GetObjectInput{
		Bucket: aws.String(bucketName),
		Key:    aws.String(objectKey),
	}

	resp, err := svc.GetObject(params)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var text strings.Builder
	jsonlReader := bufio.NewReader(resp.Body)

	firstLine := true // Flag to track the first line
	for {
		line, err := jsonlReader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return "", err
		}

		// Parse the JSONL line into a map
		var jsonObjectMap map[string]interface{}
		if err := json.Unmarshal([]byte(line), &jsonObjectMap); err != nil {
			return "", err
		}

		// Extract the "text" field
		textValue, ok := jsonObjectMap["text"].(string)
		if !ok {
			return "", fmt.Errorf("JSONL object has no 'text' field or it's not a string")
		}

		// Append the text to the result
		if firstLine {
			firstLine = false
		} else {
			text.WriteString(" ") // Append a space for all lines except the first
		}
		text.WriteString(textValue)
	}

	return text.String(), nil
}

// fetchTextFileS3 reads a text file from S3 and return its content as a string.
func fetchTextFileS3(svc S3Client, bucketName, objectKey string) (
	string,
	error,
) {
	params := &s3.GetObjectInput{
		Bucket: aws.String(bucketName),
		Key:    aws.String(objectKey),
	}

	resp, err := svc.GetObject(params)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var text strings.Builder
	textReader := bufio.NewReaderSize(resp.Body, 8*1024*1024)
	for {
		buf := make([]byte, 4096)
		n, err := textReader.Read(buf)
		if err != nil {
			if err == io.EOF {
				break
			}
			return "", err
		}
		text.Write(buf[:n])
	}

	return text.String(), nil
}

// removeS3Prefix splits the input into the bucket and to ensure that s3:// is present
func removeS3Prefix(input string) (
	hasS3Prefix bool,
	remainder string,
	s3FilePath string,
) {
	prefix := "s3://"
	if strings.HasPrefix(input, prefix) {
		//if it is just s3:// then return empty string
		if !strings.Contains(input[len(prefix):], "/") {
			return true, input[len(prefix):], ""
		}
		//if it is s3://bucket/ then return bucket and empty string
		if strings.Index(
			input[len(prefix):], "/",
		) == len(input)-len(prefix)-1 {
			return true, input[len(prefix) : len(input)-1], ""
		}
		//if it is s3://bucket/path then return bucket and path
		if strings.Index(input[len(prefix):], "/") > 0 {
			idxOfFirstSlash := strings.Index(input[len(prefix):], "/")
			bucket := input[len(prefix) : len(prefix)+idxOfFirstSlash]
			pathOfFile := input[len(prefix)+idxOfFirstSlash+1:]
			return true, bucket, pathOfFile
		}
	}
	return false, input, ""
}

// ReadTextsFromS3 reads text files recursively from all prefixes in an S3 bucket.
func ReadTextsFromS3(
	svc S3Client,
	bucketName string,
	s3FilePath string,
	sanitize bool,
	numReaderThreads int,
) (chan namedRuneReader, error) {
	runeReaders := make(chan namedRuneReader, 64)
	objects := make(chan *s3.Object, 64)
	wg := sync.WaitGroup{}

	// Start reader goroutines.
	startReader := func() {
		for {
			object, ok := <-objects
			if !ok {
				break
			}
			if s3FilePath == "" || strings.HasPrefix(
				*object.Key, s3FilePath,
			) {

				if strings.HasSuffix(*object.Key, ".jsonl") {
					// Handle JSONL files.
					jsonObject, err := fetchJSONLFileS3(
						svc, bucketName, *object.Key,
					)

					if err != nil {
						log.Printf(
							"Error reading JSONL file %s: %v", *object.Key,
							err,
						)
						continue
					}

					// Create our rune reader.
					if sanitize {
						runeReaders <- namedRuneReader{
							*object.Key,
							CreateTextSanitizer(strings.NewReader(jsonObject)),
						}
					} else {
						runeReaders <- namedRuneReader{
							*object.Key,
							strings.NewReader(jsonObject),
						}
					}
				} else {
					// Handle regular text files.
					text, err := fetchTextFileS3(svc, bucketName, *object.Key)
					if err != nil {
						log.Printf(
							"Error reading text file %s: %v", *object.Key,
							err,
						)
						continue
					}

					// Create our rune reader.
					if sanitize {
						runeReaders <- namedRuneReader{
							*object.Key,
							CreateTextSanitizer(strings.NewReader(text)),
						}
					} else {
						runeReaders <- namedRuneReader{
							*object.Key,
							strings.NewReader(text),
						}
					}
				}
			}
		}
		wg.Done()
	}

	// Start multiple reader goroutines.
	for i := 0; i < numReaderThreads; i++ {
		wg.Add(1)
		go startReader()
	}

	go func() {
		// List objects recursively.
		getObjectsS3Recursively(svc, bucketName, "", objects)

		// Close the objects channel when done.
		close(objects)

		// Wait for all reader goroutines to finish.
		wg.Wait()

		// Close the runeReaders channel.
		close(runeReaders)
	}()

	return runeReaders, nil
}

// ReadTexts
// Consumes a directory path and recursively scans for `.txt` files, producing
// a TextsIterator function that yields the text file as an io.Reader type.
func ReadTexts(
	dirPath string,
	sanitize bool,
	sortSpec string,
	numReaderThreads int,
) (chan namedRuneReader, error) {
	matches, err := GlobTexts(dirPath)
	if err != nil {
		return nil, err
	}
	if sortErr := resolveSortSpec(matches, sortSpec); sortErr != nil {
		return nil, sortErr
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
							if jErr := json.Unmarshal(
								jsonObject,
								&jsonObjectMap,
							); jErr != nil {
								log.Printf(
									"JSONL object %d in %s is not valid JSON: %s",
									idx, path.Path, jErr,
								)
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
										strings.NewReader(textString),
									)}
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
								bufio.NewReaderSize(
									fileReader,
									8*1024*1024,
								)}
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
	TokenizerId      string
	ContextSize      int
	Boundary         string
	BoundaryBegin    bool
	BoundaryOverlap  int
	PadToken         string
	EndOfText        string
	Unitrim          bool
	ExcludeTokens    []string
	SanitizeEncoding bool
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
		[]string{},
		false,
	}
}

// getAndCheckToken
// Check if s is a valid token via tokenizer table lookup; if not, we try
// encoding and checking if it is a valid single token returned.
func getAndCheckToken(
	t *gpt_bpe.GPTEncoder, s string,
	id string,
) (gpt_bpe.Token, error) {
	s = strings.ReplaceAll(s, "\\n", "\n")
	token := t.Get(s)
	if token == nil {
		tokens := *t.Encode(&s)
		// Also allow a single "real" token surrounded by an EosToken and/or a BosToken
		if len(tokens) == 1 ||
			len(tokens) == 2 && tokens[1] == t.EosToken && tokens[0] != t.BosToken {
			return tokens[0], nil
		} else if len(tokens) == 3 &&
			tokens[0] == t.BosToken && tokens[2] == t.EosToken ||
			len(tokens) == 2 && tokens[0] == t.BosToken && tokens[1] != t.EosToken {
			return tokens[1], nil
		} else {
			return 0, fmt.Errorf("'%s' is not a valid token for %s", s, id)
		}
	} else {
		return *token, nil
	}
}

func (tt *TextsTokenizer) InitTokenizer() (*gpt_bpe.GPTEncoder, error) {
	var encoderPtr *gpt_bpe.GPTEncoder
	var tokErr error
	embeddedDirName := tt.TokenizerId + "-tokenizer"
	if embedded, _ := resources.EmbeddedDirExists(embeddedDirName); embedded {
		// Check if it's an internal reference. If not, it's a file path.
		encoderPtr, tokErr = gpt_bpe.NewEncoder(embeddedDirName)
	} else {
		// Fall back to path-like.
		encoderPtr, tokErr = gpt_bpe.NewEncoder(tt.TokenizerId)
	}
	if tokErr != nil {
		return nil, tokErr
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

func (tt TextsTokenizer) handleExclusions(
	tokenizer *gpt_bpe.GPTEncoder,
) (
	err error,
) {
	if len(tt.ExcludeTokens) == 0 {
		return nil
	}
	// Remove excluded tokens from the tokenizer.
	for _, excludeToken := range tt.ExcludeTokens {
		var excludeTokenId gpt_bpe.Token
		var excludeErr error
		if excludeTokenId, excludeErr = getAndCheckToken(
			tokenizer,
			excludeToken, "ExcludeToken",
		); excludeErr != nil {
			return excludeErr
		} else {
			delete(tokenizer.Encoder, excludeToken)
			// Remove from merges.
			for i, merge := range tokenizer.TokenMerges {
				// Check if the token is in the merge, if it is, remove it
				if merge == excludeTokenId {
					mergePair := gpt_bpe.GPTPair{
						Left:  string(tokenizer.Decoder[i.Left]),
						Right: string(tokenizer.Decoder[i.Right]),
					}
					delete(tokenizer.TokenMerges, i)
					delete(tokenizer.BpeRanks, mergePair)
				}
			}
			// Remove from specials
			for i, special := range tokenizer.Specials {
				if special[0] == excludeTokenId {
					delete(tokenizer.Specials, i)
				}
			}
			tokenizer.UpdateSpecialsTree()
		}
	}
	tokenizer.UpdateSpecialsTree()
	return nil
}

func (tt TextsTokenizer) TokenizeTexts(
	texts chan namedRuneReader,
	indexPath string,
	tokenizerPtr *gpt_bpe.GPTEncoder,
) (chan gpt_bpe.Tokens, error) {
	var tokErr error
	if tokenizerPtr == nil {
		tokenizerPtr, tokErr = tt.InitTokenizer()
		if tokErr != nil {
			return nil, tokErr
		}
	}
	tokenizer := *tokenizerPtr
	var endOfText gpt_bpe.Token
	if tt.EndOfText == "" {
		endOfText = tokenizer.EosToken
	} else {
		var eotErr error
		if endOfText, eotErr = getAndCheckToken(
			&tokenizer,
			tt.EndOfText, "EndOfText",
		); eotErr != nil {
			return nil, eotErr
		}
	}

	if exclErr := tt.handleExclusions(&tokenizer); exclErr != nil {
		return nil, exclErr
	}

	if tt.SanitizeEncoding {
		tokenizer.SpecialsTree.InsertReplacementsIntoRuneTree(encodingTable)
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
				log.Printf(
					"%s tokenized in %s (%d tokens/s, %s wait)",
					runeReader.path, roundDuration,
					int(float64(tokenCt)/duration.Seconds()),
					waitDuration,
				)
				indexFile.WriteString(
					fmt.Sprintf(
						"{\"path\": \"%s\", \"offset\": %d, \"tokens\": %d}\n",
						runeReader.path, currOffset, tokenCt,
					),
				)
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
	texts chan namedRuneReader, tokenizerPtr *gpt_bpe.GPTEncoder,
) (chan gpt_bpe.Tokens, error) {
	var tokErr error
	if tokenizerPtr == nil {
		tokenizerPtr, tokErr = tt.InitTokenizer()
		if tokErr != nil {
			return nil, tokErr
		}
	}
	tokenizer := *tokenizerPtr
	var padToken, endOfText gpt_bpe.Token
	if tt.PadToken == "" {
		padToken = tokenizer.PadToken
	} else {
		var padErr error
		padToken, padErr = getAndCheckToken(
			&tokenizer, tt.PadToken,
			"PadToken",
		)
		if padErr != nil {
			return nil, padErr
		}
	}
	if tt.EndOfText == "" {
		endOfText = tokenizer.EosToken
	} else {
		var eotErr error
		endOfText, eotErr = getAndCheckToken(
			&tokenizer, tt.EndOfText,
			"EndOfText",
		)
		if eotErr != nil {
			return nil, eotErr
		}
	}

	var boundary gpt_bpe.Token
	if tt.Boundary == "" {
		boundary = 0xFFFFFFFF
	} else {
		var boundaryErr error
		boundary, boundaryErr = getAndCheckToken(
			&tokenizer, tt.Boundary,
			"Boundary",
		)
		if boundaryErr != nil {
			return nil, boundaryErr
		}
	}
	contextSize := tt.ContextSize
	doUnitrim := tt.Unitrim

	if exclErr := tt.handleExclusions(&tokenizer); exclErr != nil {
		return nil, exclErr
	}

	if tt.SanitizeEncoding {
		tokenizer.SpecialsTree.InsertReplacementsIntoRuneTree(encodingTable)
	}

	var tokens gpt_bpe.Tokens
	var done bool
	var numTokens, idx, begin int
	boundaryIdxes := make([]int, 0)

	// Consume texts from `nextText()` and tokenize as a `goroutine`.
	tokenizedTexts := make(chan gpt_bpe.Tokens, 16)
	contexts := make(chan gpt_bpe.Tokens, 16)
	nextTokenized := func() {
		tokenCt := 0
		for {
			waitBegin := time.Now()
			runeReader, more := <-texts
			beginTs := time.Now()
			if more {
				wait_duration := time.Since(waitBegin)
				encodeChunk := tokenizer.StreamingEncode(runeReader.reader)
				for {
					tokenized := encodeChunk(contextSize * 8)
					if tokenized == nil {
						duration := time.Since(beginTs)
						// If we took longer than a millisecond, round to the
						// nearest  millisecond.
						var roundDuration time.Duration
						if duration.Round(time.Millisecond) > 0 {
							roundDuration = duration.Round(time.Millisecond)
						} else {
							roundDuration = duration
						}
						tokenizedTexts <- gpt_bpe.Tokens{endOfText}
						log.Printf(
							"%s tokenized in %s (%d tokens/s, %s wait)",
							runeReader.path, roundDuration,
							int(float64(tokenCt)/duration.Seconds()),
							wait_duration,
						)
						tokenCt = 0
						beginTs = time.Now()
						break
					}
					tokenizedTexts <- *tokenized
					tokenCt += len(*tokenized)
				}
			} else {
				break
			}
		}
		close(tokenizedTexts)
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
						chunk, endAt = tokenizer.AlignAndSizeTokens(
							&chunk,
							contextSize,
						)
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
					if boundary == 0xFFFFFFFF && doUnitrim {
						// Ensure that our next chunk is aligned to valid
						// unicode.
						_, offset := tokenizer.AlignAndSizeTokens(
							&chunk,
							idx-begin,
						)
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
			if context == nil || len(*context) == 0 {
				break
			} else {
				contexts <- *context
			}
		}
		close(contexts)
	}()

	return contexts, nil
}

// GCD, Greatest Common Divisor of two integers.
func GCD(a, b int) int {
	for b != 0 {
		t := b
		b = a % b
		a = t
	}
	return a
}

// WriteContexts
// Consumes a ContextsIterator function and serializes the contexts to an
// aligned binary file.
func WriteContexts(
	outPath string,
	contexts chan gpt_bpe.Tokens,
	encoder *gpt_bpe.GPTEncoder,
	sampling int,
	shuffle bool,
	enforceUint32 bool,
	showContexts bool,
) (int, error) {
	totalTokens := 0
	useUint32 := enforceUint32
	// Use uint32 if explicitly requested or if the vocab size is greater than 65536.
	if !useUint32 {
		if encoder == nil {
			return 0, fmt.Errorf("WriteContexts called with unknown encoder; cannot determine output byte width")
		} else if len(encoder.Encoder) > 65536 {
			useUint32 = true
			log.Println("warning: tokenizer vocab too large for 16-bit, outputting as 32-bit")
		}
	}
	if showContexts && encoder == nil {
		showContexts = false
		log.Println("warning: no encoder info, cannot show contexts")
	}

	// create file AND filepath if not exists
	if err := os.MkdirAll(filepath.Dir(outPath), os.ModePerm); err != nil {
		return 0, err
	}
	outFile, err := os.OpenFile(
		outPath, os.O_TRUNC|os.O_RDWR|os.O_CREATE,
		0755,
	)
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
				// If sampling is 80, GCD is 20, LCD is 5, 80% of 5 is 4
				// So we keep 4 and skip 5th
				gcdOfSampling := GCD(sampling, 100)
				lcd := 100 / gcdOfSampling
				samplingFloat := float64(sampling) / 100.0
				skipEveryX := int(math.Round(samplingFloat * float64(lcd)))
				doKeepSampling := sampling == 100 || (samplingIdx%lcd < skipEveryX)
				if doKeepSampling {
					sampledContexts <- context
					if showContexts {
						fmt.Println(len(context))
						fmt.Println("======================================")
						fmt.Println(encoder.Decode(&context))
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
	var prevTarget int64

	// Sometimes it is requested that we shuffle all contexts as they are
	// written to the file. This is useful for training data, as it can help
	// the model learn better.

	// We keep track of the idxes of the padding we add to the contexts
	type paddingTuple struct {
		contextStart int
		contextEnd   int
		start        int
		end          int
	}
	idxes := make([]paddingTuple, 0)
	for {
		context, ok := <-sampledContexts
		if !ok {
			break
		}
		binContext, err := context.ToBin(useUint32)
		if err != nil {
			return totalTokens, err
		}
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
			_, err := outFile.ReadAt(buf, target)
			if err != nil {
				return totalTokens, err
			}

		} else if shuffle {
			//write the buffer to the end of the file
			if _, err := outFile.Write(*binContext); err != nil {
				return totalTokens, err
			}
		}

		if endpos > 0 && shuffle {
			// Pad to context size if necessary
			if len(*binContext) < contextSize {
				lenBeforePad := len(*binContext)
				for i := len(*binContext); i < contextSize; i++ {
					(*binContext) = append(*binContext, byte(0))
				}
				// Move the paddingTuple to new position if necessary
				for i, idx := range idxes {
					if idx.contextStart == int(prevTarget) &&
						idx.contextEnd == int(prevTarget)+contextSize {
						// Move to end of file
						idxes[i].contextStart = endpos
						idxes[i].contextEnd = endpos + contextSize
					}
				}

				idxes = append(
					idxes, paddingTuple{start: lenBeforePad,
						end: contextSize, contextStart: int(target),
						contextEnd: int(target) + contextSize},
				)
			}
			// Overwrite binContext to the location of the context we just read
			_, err := outFile.WriteAt(*binContext, target)
			if err != nil {
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

		// Break if EOF
		if len(context) <= 1 {
			break
		}
	}

	// Write new file with padding removed
	if shuffle {
		newFilePath := strings.Replace(outPath, ".chunk", ".shuf.chunk", 1)
		newFile, err := os.OpenFile(
			newFilePath, os.O_TRUNC|os.O_RDWR|os.O_CREATE, 0755,
		)
		defer newFile.Close()
		if err != nil {
			return totalTokens, err
		}
		// Get contexts from original file, skipping idxes
		buf := make([]byte, contextSize)
		outFile.Seek(0, 0)
		currentStartPos := 0
		currentEndPos := currentStartPos + contextSize
		for {
			num, err := outFile.ReadAt(buf, int64(currentStartPos))
			if err != nil {
				if err == io.EOF {
					break
				}
				return totalTokens, err
			}
			if num == 0 {
				break
			}
			thisContext := paddingTuple{contextStart: currentStartPos,
				contextEnd: currentEndPos, start: 0, end: len(buf)}
			// Check if the context is in the idxes
			for _, idx := range idxes {
				if idx.contextStart == thisContext.contextStart &&
					idx.contextEnd == thisContext.contextEnd {
					if idx.start < thisContext.start || thisContext.start == 0 {
						thisContext.start = idx.start
					}
					if idx.end > thisContext.end || thisContext.end == 0 {
						thisContext.end = idx.end
					}
				}
			}
			if thisContext.start != 0 && thisContext.end != 0 {
				newFile.Write(buf[:thisContext.start])

			} else {
				newFile.Write(buf)
			}
			currentStartPos = currentEndPos
			currentEndPos = currentStartPos + contextSize
		}
		// Replace the original file with the new file
		if err := os.Rename(newFilePath, outPath); err != nil {
			return totalTokens, err
		}
	}

	return totalTokens, nil
}

func main() {
	tokenizerId := flag.String(
		"tokenizer", "gpt2",
		"tokenizer to use [gpt2, pile, nerdstash_v1, "+
			"nerdstash_v2, huggingface-id]",
	)
	contextSize := flag.Int("context", 2048, "context size")
	showContexts := flag.Bool(
		"show_contexts", false,
		"show contexts as they are tokenized",
	)
	endOfText := flag.String(
		"eot", "",
		"end of text token to split texts, can be token or int16 "+
			"token_id",
	)
	padToken := flag.String(
		"pad", "",
		"pad token to pad out contexts, can be <|padding|>, or an "+
			"int16 token_id",
	)
	boundaryToken := flag.String(
		"boundary", "\n",
		"boundary token to split contexts on, can be a string token "+
			"or int16 token_id",
	)
	boundaryBegin := flag.Bool(
		"boundary_begin", false,
		"whether to treat the boundary token as a beginning token for"+
			"a context",
	)
	boundaryOverlap := flag.Int(
		"boundary_overlap", -1,
		"token index in context to approximately overlap contexts on, "+
			"-1 for last boundary token in context",
	)
	outputFile := flag.String(
		"output", "tokenized.chunk",
		"tokenized output file",
	)
	inputDir := flag.String(
		"input", "",
		"input directory",
	)
	unitrimBool := flag.Bool(
		"no_unitrim", false,
		"do not trim contexts to valid unicode",
	)
	forceRetokenization := flag.Bool(
		"retokenize", false,
		"force retokenization even if tokenizer output is newer",
	)
	sanitizeBool := flag.Bool(
		"sanitize", false,
		"sanitize inputs of whitespace issues",
	)
	reorderPaths := flag.String(
		"reorder", "",
		"reorder input files to specification [size_ascending, "+
			"size_descending, name_ascending, name_descending, random, shuffle, none]",
	)
	sampling_str := flag.String(
		"sampling", "100", "a integer value from 0-100 "+
			"which tells the tokenizer how many chunks to discard in %, 60 keeps 60%% chunks",
	)
	streaming_encode := flag.Bool(
		"streaming_encode", false,
		"use streaming encode, which writes to disk as it encodes, "+
			"rather than buffering into contexts",
	)
	numThreads := flag.Int(
		"threads", 4,
		"number of tokenization threads to use",
	)
	numReaderThreads := flag.Int(
		"reader_threads", 2,
		"number of I/O reader threads to use",
	)
	excludeTokens := flag.String(
		"exclude_tokens", "",
		"comma separated list of tokens to exclude from the vocabulary",
	)
	sanitizeEncodingBool := flag.Bool(
		"disable_sanitize_encoding",
		false, "disable sanitizing of misencoding",
	)
	s3Endpoint := flag.String(
		"object_storage_endpoint", "https://object.las1.coreweave.com",
		"CW S3 Endpoint to use for fetching data",
	)
	enforceUint32 := flag.Bool(
		"uint32_enforce", false,
		"output tokens as uint32 instead of uint16 (required for vocabs with over 2^16 tokens)",
	)

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
	log.Printf(
		"Sampling amount (in %s tokens kept): %d%s\n",
		"%", sampling, "%",
	)

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

	var excludeTokensList []string
	if *excludeTokens != "" {
		excludeTokensList = strings.Split(*excludeTokens, ",")
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
	textsTokenizer.ExcludeTokens = excludeTokensList
	textsTokenizer.SanitizeEncoding = !*sanitizeEncodingBool

	if !*forceRetokenization {
		if outStat, outErr := os.Stat(*outputFile); !errors.Is(
			outErr,
			os.ErrNotExist,
		) && outErr != nil {
			log.Fatal(outErr)
		} else if errors.Is(outErr, os.ErrNotExist) {
			log.Printf("Creating %s", *outputFile)
		} else if newestPath, newestModTime, newestErr := FindNewestText(
			*inputDir,
		); newestErr != nil {
			log.Fatal(newestErr)
		} else if newestModTime != nil && newestModTime.Before(
			outStat.ModTime(),
		) {
			log.Printf(
				"Newest source `%s` is older than `%s`, "+
					"not retokenizing. "+
					"Use -retokenize to force retokenization.", *newestPath,
				*outputFile,
			)
			os.Exit(0)
		} else if newestDir, newestDirModTime, newestDirErr := FindNewestDir(
			*inputDir,
		); newestDirErr != nil {
			log.Fatal(newestDirErr)
		} else if newestDirModTime != nil && newestDirModTime.Before(
			outStat.ModTime(),
		) {
			log.Printf(
				"Data source directory `%s` has no changes since `%s"+
					"was tokenized. Use -retokenize to force retokenization.",
				*newestDir, *outputFile,
			)
		}
	}

	encoder, tokErr := textsTokenizer.InitTokenizer()
	if tokErr != nil {
		log.Fatal(tokErr)
	}

	hasS3Prefix, s3Bucket, s3FilePath := removeS3Prefix(*inputDir)

	if hasS3Prefix && *s3Endpoint == "" {
		flag.Usage()
		log.Fatal("Must provide S3 Endpoint if fetching data from CW object storage")
	}

	// Declare textReaders
	var textReaders chan namedRuneReader

	if hasS3Prefix && *s3Endpoint != "" {
		defaultResolver := endpoints.DefaultResolver()
		s3CustResolverFn := func(
			service, region string,
			optFns ...func(*endpoints.Options),
		) (endpoints.ResolvedEndpoint, error) {
			if service == "s3" {
				return endpoints.ResolvedEndpoint{
					URL: *s3Endpoint,
				}, nil
			}

			return defaultResolver.EndpointFor(service, region, optFns...)
		}

		sess := session.Must(
			session.NewSessionWithOptions(
				session.Options{
					Config: aws.Config{
						EndpointResolver: endpoints.ResolverFunc(s3CustResolverFn),
						Region:           aws.String("coreweave-object-storage"),
					},
				},
			),
		)

		svc := s3.New(sess)
		textReaders, err = ReadTextsFromS3(
			svc, s3Bucket, s3FilePath, *sanitizeBool, *numReaderThreads,
		)

		if err != nil {
			log.Fatal(err)
		}
	} else {
		textReaders, err = ReadTexts(
			*inputDir, *sanitizeBool, *reorderPaths, *numReaderThreads,
		)

		if err != nil {
			log.Fatal(err)
		}
	}

	numTokens := 0
	begin := time.Now()
	if *streaming_encode {
		wg := sync.WaitGroup{}
		for threadIdx := 0; threadIdx < *numThreads; threadIdx++ {
			wg.Add(1)
			go func(threadId int) {
				var contexts chan gpt_bpe.Tokens
				var tokErr error
				indexFilePath := fmt.Sprintf(
					"%s.%d.index",
					*outputFile, threadId,
				)
				outputFilePath := fmt.Sprintf(
					"%s.%d.tokens",
					*outputFile, threadId,
				)
				contexts, tokErr = textsTokenizer.TokenizeTexts(
					textReaders,
					indexFilePath,
					encoder,
				)
				if tokErr != nil {
					log.Fatal(tokErr)
				}
				total, writeErr := WriteContexts(
					outputFilePath,
					contexts,
					encoder,
					sampling,
					false,
					*enforceUint32,
					*showContexts,
				)
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
			textReaders, encoder,
		)
		if tokErr != nil {
			log.Fatal(tokErr)
		}
		var writeErr error
		numTokens, writeErr = WriteContexts(
			*outputFile,
			contexts,
			encoder,
			sampling,
			*reorderPaths == "shuffle",
			*enforceUint32,
			*showContexts,
		)
		if writeErr != nil {
			log.Fatal(writeErr)
		}
	}

	duration := time.Since(begin).Seconds()

	log.Printf(
		"%d tokens in %0.2fs, %0.2f tokens/s", numTokens,
		duration, float64(numTokens)/duration,
	)
}

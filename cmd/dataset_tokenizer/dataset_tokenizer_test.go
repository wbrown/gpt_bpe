package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/stretchr/testify/assert"
	"github.com/wbrown/gpt_bpe"
)

type SanitizerTest struct {
	Name     string
	Input    string
	Expected string
}

// S3MockClient is a mock implementation of S3Client.
type S3MockClient struct {
	GetObjectOutput     *s3.GetObjectOutput
	GetObjectError      error
	GetObjectOutputs    map[string]*s3.GetObjectOutput // Map object keys to GetObjectOutput
	GetObjectErrors     map[string]error
	ListObjectsV2Output *s3.ListObjectsV2Output
	ListObjectsV2Error  error
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

const corpusPath = "../../resources/frankenstein.txt"

func TokensFromBin(bin *[]byte) *gpt_bpe.Tokens {
	tokens := make(gpt_bpe.Tokens, 0)
	buf := bytes.NewReader(*bin)
	for {
		var token gpt_bpe.Token
		if err := binary.Read(buf, binary.LittleEndian, &token); err != nil {
			break
		}
		tokens = append(tokens, token)
	}
	return &tokens
}

// DecodeBuffer
// Decode Tokens from a byte array into a string.
func DecodeBuffer(encoded *[]byte) (text string) {
	// First convert our bytearray into a uint32 `Token` array.
	tokens := TokensFromBin(encoded)
	// Decode our tokens into a string.
	var enc *gpt_bpe.GPTEncoder
	encoderString := "gpt2"
	enc, _ = gpt_bpe.NewEncoder(encoderString)
	return enc.Decode(tokens)
}

func BenchmarkSanitizeText(b *testing.B) {
	b.StopTimer()
	b.ResetTimer()
	path := corpusPath
	if testFile, err := os.Open(path); err != nil {
		b.Fail()
	} else {
		b.StartTimer()
		reader := CreateTextSanitizer(testFile)
		runes := make([]rune, 0)
		for {
			r, size, _ := reader.ReadRune()
			if size > 0 {
				runes = append(runes, r)
			} else {
				break
			}
		}
		b.StopTimer()
		b.Logf("%d runes read", len(runes))
	}
}

func BenchmarkStreamingEncode(b *testing.B) {
	b.StopTimer()
	b.ResetTimer()
	tokenizer, err := gpt_bpe.NewEncoder("nerdstash_v1")
	if err != nil {
		log.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		if testFile, err := os.Open(corpusPath); err != nil {
			b.Fail()
		} else {
			start := time.Now()
			b.StartTimer()
			nextChunk := tokenizer.StreamingEncode(bufio.NewReader(testFile))
			tokensCt := 0
			for {
				if chunk := nextChunk(2048); chunk == nil {
					break
				} else {
					tokensCt += len(*chunk)
				}
			}
			b.StopTimer()
			duration := time.Now().Sub(start)
			tokensPerSecond := float64(tokensCt) / duration.Seconds()
			lruStats := fmt.Sprintf(" (LRU: Hits: %d, Misses: %d, "+
				"Evictions: %d, %0.2f%% Hit Rate, Size: %d)",
				tokenizer.LruHits, tokenizer.LruMisses,
				tokenizer.LruEvictions, 100.0*float64(
					tokenizer.LruHits)/float64(tokenizer.LruHits+
					tokenizer.LruMisses), tokenizer.LruSize)
			b.Logf("%d tokens generated at %0.2f per second over %vms%s",
				tokensCt,
				tokensPerSecond, duration.Milliseconds(),
				lruStats)
		}
	}
}

func BenchmarkStreamingEncodeSanitize(b *testing.B) {
	b.StopTimer()
	b.ResetTimer()
	tokenizer, err := gpt_bpe.NewEncoder("gpt2")
	if err != nil {
		log.Fatal(err)
	}
	path := corpusPath
	if testFile, err := os.Open(path); err != nil {
		b.Fail()
	} else {
		start := time.Now()
		b.StartTimer()
		reader := CreateTextSanitizer(testFile)
		nextChunk := tokenizer.StreamingEncode(reader)
		tokensCt := 0
		for {
			if chunk := nextChunk(2048); chunk == nil {
				break
			} else {
				tokensCt += len(*chunk)
			}
		}
		b.StopTimer()
		tokensPerSecond := float64(tokensCt) / time.Now().Sub(start).Seconds()
		b.Logf("%d tokens generated at %0.2f per second", tokensCt,
			tokensPerSecond)
	}
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

func TestSampling40(t *testing.T) {
	all1 := 0
	all2 := 0

	textsTokenizer := NewTextsTokenizer()
	textsTokenizer.ContextSize = 2048
	textsTokenizer.TokenizerId = "gpt2"
	textsTokenizer.EndOfText = ""
	textsTokenizer.PadToken = ""
	textsTokenizer.Boundary = "\n"
	textsTokenizer.Unitrim = true
	textsTokenizer.BoundaryBegin = false

	inputDir := "../../resources"
	reorderPaths := ""
	sampling := 100
	outputFile := "base.chunk"

	if _, tokErr := textsTokenizer.InitTokenizer(); tokErr != nil {
		log.Fatal(tokErr)
	}

	if texts, err := ReadTexts(inputDir, false,
		reorderPaths,
		1); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts, tokErr := textsTokenizer.TokenizeTexts(
			texts, "test")
		if tokErr != nil {
			log.Fatal(tokErr)
		}

		var enc *gpt_bpe.GPTEncoder
		// *showContexts = true

		total, writeErr := WriteContexts(outputFile, contexts, enc, sampling, reorderPaths == "shuffle")
		all1 += total
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", total,
			duration, float64(total)/duration)
	}

	textsTokenizer2 := NewTextsTokenizer()
	textsTokenizer2.ContextSize = 2048
	textsTokenizer2.TokenizerId = "gpt2"
	textsTokenizer2.EndOfText = ""
	textsTokenizer2.PadToken = ""
	textsTokenizer2.Boundary = "\n"
	textsTokenizer2.Unitrim = true
	textsTokenizer2.BoundaryBegin = false

	inputDir = "../../resources"
	reorderPaths = ""
	sampling = 40
	outputFile = "samp40.chunk"

	if _, tokErr := textsTokenizer.InitTokenizer(); tokErr != nil {
		log.Fatal(tokErr)
	}

	if texts2, err := ReadTexts(inputDir, false,
		reorderPaths,
		1); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts, tokErr := textsTokenizer.TokenizeTexts(
			texts2, "")
		if tokErr != nil {
			log.Fatal(tokErr)
		}
		var enc *gpt_bpe.GPTEncoder
		// *showContexts = true

		total2, writeErr := WriteContexts(outputFile, contexts, enc, sampling, reorderPaths == "shuffle")
		all2 += total2
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", total2,
			duration, float64(total2)/duration)
	}
	percent := (float64(all2) / float64(all1)) * 100
	log.Printf("Sampling 100 produced %d Tokens, Sampling 40 produced %d tokens\n", all1, all2)
	log.Printf("Roughly %f %%\n", percent)
	if percent > 55 || percent < 25 {
		log.Printf("Percent does not match ~40%% (25-55), found to be %f", percent)
		t.Fail()
	}
}

func TestShuffle(t *testing.T) {
	all1 := 0
	all2 := 0

	textsTokenizer := NewTextsTokenizer()
	textsTokenizer.ContextSize = 2048
	textsTokenizer.TokenizerId = "gpt2"
	textsTokenizer.EndOfText = ""
	textsTokenizer.PadToken = "_"
	textsTokenizer.Boundary = "\n"
	textsTokenizer.Unitrim = true
	textsTokenizer.BoundaryBegin = false

	inputDir := "../../resources"
	reorderPaths := ""
	sampling := 100
	outputFile := "noshuffle.chunk"

	if _, tokErr := textsTokenizer.InitTokenizer(); tokErr != nil {
		log.Fatal(tokErr)
	}

	if texts, err := ReadTexts(inputDir, true,
		reorderPaths, 1); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts, tokErr := textsTokenizer.TokenizeTexts(
			texts, "test")
		if tokErr != nil {
			log.Fatal(tokErr)
		}
		var enc *gpt_bpe.GPTEncoder
		// *showContexts = true

		total, writeErr := WriteContexts(outputFile, contexts, enc, sampling, reorderPaths == "shuffle")
		all1 += total
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", total,
			duration, float64(total)/duration)
	}

	textsTokenizer2 := NewTextsTokenizer()
	textsTokenizer2.ContextSize = 2048
	textsTokenizer2.TokenizerId = "gpt2"
	textsTokenizer2.EndOfText = ""
	textsTokenizer2.PadToken = "_"
	textsTokenizer2.Boundary = "\n"
	textsTokenizer2.Unitrim = true
	textsTokenizer2.BoundaryBegin = false

	inputDir = "../../resources"
	reorderPaths = "shuffle"
	sampling = 100
	outputFile = "shuffle.chunk"

	if _, tokErr := textsTokenizer.InitTokenizer(); tokErr != nil {
		log.Fatal(tokErr)
	}

	if texts2, err := ReadTexts(inputDir, true,
		reorderPaths, 1); err != nil {
		log.Fatal(err)
	} else {
		begin := time.Now()
		contexts2, tokErr := textsTokenizer.TokenizeTexts(
			texts2, "test")
		if tokErr != nil {
			log.Fatal(tokErr)
		}
		var enc2 *gpt_bpe.GPTEncoder
		// *showContexts = true

		total2, writeErr := WriteContexts(outputFile, contexts2, enc2, sampling, reorderPaths == "shuffle")
		all2 += total2
		if writeErr != nil {
			log.Fatal(writeErr)
		}
		duration := time.Now().Sub(begin).Seconds()
		log.Printf("%d tokens in %0.2fs, %0.2f tokens/s", total2,
			duration, float64(total2)/duration)
	}
	percent := (float64(all2) / float64(all1)) * 100
	log.Printf("NoShuffle produced %d Tokens, Shuffle produced %d tokens, this is roughly %f %%", all1, all2, percent)
	if percent != 100 {
		log.Printf("Percent does not match ~100%%, found to be %f", percent)
		t.Fail()
	}

	f, err := os.Open("noshuffle.chunk")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	f2, err2 := os.Open("shuffle.chunk")
	if err2 != nil {
		log.Fatal(err2)
	}
	defer f2.Close()
	//chunk by chunk verification of shuffle

	f.Seek(0, 0)
	var verifymap = make(map[string]bool)
	buffer := make([]byte, 4096)
	for {
		bytesread, err := f.Read(buffer)
		//break up into tokens
		if err != nil {
			if err != io.EOF {
				fmt.Println(err)
			}
			break
		}
		hash := sha256.Sum256(buffer[0 : bytesread-1])
		sha := hex.EncodeToString(hash[:])
		verifymap[sha] = false

	}
	f2.Seek(0, 0)
	buffer2 := make([]byte, 4096)
	for {
		bytesread2, err2 := f2.Read(buffer2)
		if err2 != nil {
			if err2 != io.EOF {
				fmt.Println(err2)
			}
			break
		}

		hash2 := sha256.Sum256(buffer2[0 : bytesread2-1])
		sha2 := hex.EncodeToString(hash2[:])
		//fmt.Printf("SHA256: %s", sha2)
		//slice2 := buffer2[0 : bytesread2-1]
		//fmt.Printf("STARTHERE-->%v\n", TokensFromBin(&slice2))

		if _, ok := verifymap[sha2]; ok {
			verifymap[sha2] = true
		}

	}

	for key, value := range verifymap {
		if value == false {
			fmt.Printf("Failed at %s\n", key)
			t.Fail()
		}
	}

	fmt.Printf("Using Chunk by chunk hashing, shuffle found to be working as intended!! \n")
}

func (m *S3MockClient) ListObjectsV2(input *s3.ListObjectsV2Input) (*s3.ListObjectsV2Output, error) {
	return m.ListObjectsV2Output, m.ListObjectsV2Error
}

func (m *S3MockClient) GetObject(input *s3.GetObjectInput) (*s3.GetObjectOutput, error) {
	return m.GetObjectOutput, m.GetObjectError
}

func (m *S3MockClient) GetObjects(bucketName, prefix string) ([]*s3.Object, error) {
	// Simulate listing objects with the specified prefix
	var matchingObjects []*s3.Object

	for key := range m.GetObjectOutputs {
		if strings.HasPrefix(key, prefix) {
			matchingObjects = append(matchingObjects, &s3.Object{Key: aws.String(key)})
		}
	}

	if len(matchingObjects) == 0 {
		// Simulate the case where no objects match the prefix
		return nil, awserr.New("NoSuchKey", "The specified key does not exist", nil)
	}

	return matchingObjects, nil
}

func TestFetchJSONLFileS3(t *testing.T) {
	// Define a JSONL file content for testing
	jsonlContent := `{"text": "Hello, World!"}
{"text": "Testing JSONL"}
{"text": "This line should be valid"}
`

	// Create a mock S3 client
	mockSvc := &S3MockClient{
		GetObjectOutput: &s3.GetObjectOutput{
			Body:          io.NopCloser(strings.NewReader(jsonlContent)),
			ContentLength: aws.Int64(int64(len(jsonlContent))),
		},
		GetObjectError: nil, // No error for this test
	}

	// Call readJSONLFileS3 with the mock S3 client
	text, err := fetchJSONLFileS3(mockSvc, "test-bucket", "test-object.jsonl")

	if err != nil {
		t.Errorf("Expected no error, but got %v", err)
	}

	// Case 1: Verify that the extracted text matches the expected result
	expectedText := "Hello, World! Testing JSONL This line should be valid"
	if text != expectedText {
		t.Errorf("Expected text: %s but got: %s", expectedText, text)
	}

	// Case 2: Test case with an error returned by GetObject
	mockSvc.GetObjectError = errors.New("Simulated error")
	_, err = fetchJSONLFileS3(mockSvc, "test-bucket", "error-object.jsonl")

	if err == nil {
		t.Error("Expected an error, but got none")
	}
}

func TestFetchTextFileS3(t *testing.T) {
	// Define test data
	textContent := "This is a test. This is great. Have fun in life."

	// Create a mock S3 client
	mockSvc := &S3MockClient{
		GetObjectOutput: &s3.GetObjectOutput{
			Body:          io.NopCloser(strings.NewReader(textContent)),
			ContentLength: aws.Int64(int64(len(textContent))),
		},
		GetObjectError: nil, // No error for this test
	}

	// Call readTextFileS3 with the mock S3 client
	text, err := fetchTextFileS3(mockSvc, "test-bucket", "test-object.txt")

	if err != nil {
		t.Errorf("Expected no error, but got %v", err)
	}

	// Case 1: Verify that the extracted text matches the expected result
	if text != textContent {
		t.Errorf("Expected text: %s, but got: %s", textContent, text)
	}

	// Cas 2: Test case with an error returned by GetObject
	mockSvc.GetObjectError = errors.New("Simulated error")
	_, err = fetchTextFileS3(mockSvc, "test-bucket", "error-object.txt")

	if err == nil {
		t.Error("Expected an error, but got none")
	}
}

func TestListObjectsRecursively(t *testing.T) {
	// Create a mock S3 client
	mockSvc := &S3MockClient{
		// Define the expected behavior for ListObjectsV2
		ListObjectsV2Output: &s3.ListObjectsV2Output{
			Contents: []*s3.Object{
				{Key: aws.String("a/b/c/d/object1.txt")},
				{Key: aws.String("b/c/d/object2.jsonl")},
			},
			IsTruncated:           aws.Bool(false),
			NextContinuationToken: nil,
		},
	}

	// Create a channel for objects
	objects := make(chan *s3.Object, 10)

	// Create a WaitGroup
	var wg sync.WaitGroup
	wg.Add(1)

	// Call listObjectsRecursively with the mock S3 client
	go func() {
		defer wg.Done()
		getObjectsS3Recursively(mockSvc, "test-bucket", "prefix/", objects)
		close(objects) // Close the channel when finished
	}()

	// Receive objects from the channel
	var receivedObjects []*s3.Object
	for obj := range objects {
		receivedObjects = append(receivedObjects, obj)
	}

	// Verify the number of received objects
	if len(receivedObjects) != 2 {
		t.Errorf("Expected 2 objects, but got %d", len(receivedObjects))
	}

	// Verify the keys of received objects
	expectedKeys := []string{"a/b/c/d/object1.txt", "b/c/d/object2.jsonl"}
	for i, obj := range receivedObjects {
		if *obj.Key != expectedKeys[i] {
			t.Errorf("Expected key %s, but got %s", expectedKeys[i], *obj.Key)
		}
	}

	wg.Wait() // Wait for all goroutines to finish
}

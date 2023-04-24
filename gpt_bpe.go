package gpt_bpe

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"

	lru "github.com/hashicorp/golang-lru"
	"github.com/wbrown/gpt_bpe/resources"
)

const BPE_LRU_SZ = 65536
const RUNEBUF_SZ = 16384
const WORDCHAN_SZ = 4096

type Token uint16
type Tokens []Token

type GPTEncoder struct {
	Encoder         map[string]Token
	Decoder         map[Token][]byte
	BpeRanks        map[GPTPair]float64
	TokenMerges     map[TokenPair]Token
	BytesEncoder    *map[byte]Token
	unitrim         []int
	pattern         *regexp.Regexp
	puncPat         *regexp.Regexp
	specialsPat     *regexp.Regexp
	byteToRune      [256]rune
	runeToByte      map[rune]byte
	Specials        map[string]Tokens
	specialsArr     []string
	specialsTree    *RuneNode
	Cache           *lru.ARCCache
	PuncRunes       []rune
	Normalizer      *strings.Replacer
	DecodeExtra     *strings.Replacer
	BosToken        Token
	EosToken        Token
	PadToken        Token
	encloseEosBos   bool
	prefixSpace     bool
	lowerCase       bool
	endOfWord       string
	replacements    map[string]string
	runeBufSz       int
	wordChanSz      int
	LruHits         int
	LruMisses       int
	LruEvictions    int
	LruSize         int
	SplitterThreads int
}

type GPTPair struct {
	Left  string
	Right string
}

type TokenPair struct {
	Left  Token
	Right Token
}

type BGERank struct {
	rank   float64
	bigram GPTPair
}

type BGERanks []BGERank

func (bs BGERanks) Len() int {
	return len(bs)
}

func (bs BGERanks) Swap(i, j int) {
	bs[i], bs[j] = bs[j], bs[i]
}

func (bs BGERanks) Less(i, j int) bool {
	return bs[i].rank < bs[j].rank
}

const SPLIT_REGEX = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L" +
	"}+| ?\\p{N}+| ?[^\\s\\p{L" +
	"}\\p{N}]+|\\s+(\\S){0}|\\s+"
const PUNC_REGEX = "\\p{L}[.!?;]\\p{L}"
const REGEX_ERROR = "gpt_bpe: Fatal error compiling regular expression: %v"

func NewGPT2Encoder() GPTEncoder {
	encoder, _ := NewEncoder("gpt2-tokenizer")
	return *encoder
}

func NewPileEncoder() GPTEncoder {
	encoder, _ := NewEncoder("pile-tokenizer")
	return *encoder
}

func NewCLIPEncoder() GPTEncoder {
	encoder, _ := NewEncoder("clip-tokenizer")
	return *encoder
}

func NewNerdstashEncoder() GPTEncoder {
	encoder, _ := NewEncoder("nerdstash-tokenizer")
	return *encoder
}

// NewEncoder
// Returns a GPTEncoder with the tokenizer data loaded for that vocabulary
// id.
func NewEncoder(vocabId string) (*GPTEncoder, error) {
	hfConfig, resourcesPtr, vocabErr := resources.ResolveVocabId(vocabId,
		"")
	if vocabErr != nil {
		return nil, vocabErr
	}
	rsrcs := *resourcesPtr

	if hfConfig != nil && hfConfig.ModelId != nil {
		vocabId = *hfConfig.ModelId
	}

	specialConfig := resources.SpecialConfig{
		PuncRunes:     nil,
		Normalizer:    nil,
		EncloseEosBos: false,
		PrefixSpace:   true,
		LowerCase:     false,
		EndOfWord:     "",
		DecodeExtra:   nil,
		SplitRegex:    nil,
	}
	if special, ok := (rsrcs)["special_config.json"]; ok {
		if special.Data != nil {
			if json.Unmarshal(*special.Data, &specialConfig) != nil {
				log.Fatal("Error unmarshalling special_config.json")
			}
		}
	}
	puncRunes := make([]rune, 0)
	if specialConfig.PuncRunes != nil {
		for _, r := range specialConfig.PuncRunes {
			puncRunes = append(puncRunes, rune((*r)[0]))
		}
	}
	normalizer := strings.NewReplacer()
	if specialConfig.Normalizer != nil {
		norms := make([]string, 0)
		for k, v := range *specialConfig.Normalizer {
			norms = append(norms, string(k), string(v))
		}
		normalizer = strings.NewReplacer(norms...)
	}
	decodeExtra := strings.NewReplacer()
	if specialConfig.DecodeExtra != nil {
		decode := make([]string, 0)
		for k, v := range *specialConfig.DecodeExtra {
			decode = append(decode, string(k), string(v))
		}
		decodeExtra = strings.NewReplacer(decode...)
	}

	// Unmarshal the Encoder mappings from json to (int: string) map.
	encoderMappings := make(map[string]Token)
	// check if the Encoder.json file is present
	if _, ok := rsrcs["encoder.json"]; !ok {
		return nil, fmt.Errorf("encoder.json not found for vocabId: %s",
			vocabId)
	}
	if json.Unmarshal(*rsrcs["encoder.json"].Data, &encoderMappings) != nil {
		log.Fatal("Error unmarshalling `Encoder.json`")
	}

	// Build the unitrim array dynamically.
	unitrimArr := makeUnitrimArr(encoderMappings)

	// Build the bytes to unicode tables.
	bytesUnicodeMap := make(map[byte]rune)
	unicodeBytes := make(map[rune]byte)
	for b := uint8('!'); b < uint8('~')+1; b++ {
		bytesUnicodeMap[b] = rune(b)
		unicodeBytes[rune(b)] = b
	}
	for b := uint8('¡'); b < uint8('¬')+1; b++ {
		bytesUnicodeMap[b] = rune(b)
		unicodeBytes[rune(b)] = b
	}
	for b := uint16('®'); b < uint16('ÿ')+1; b++ {
		bytesUnicodeMap[byte(b)] = rune(b)
		unicodeBytes[rune(b)] = byte(b)
	}
	uct := 0
	var bytesUnicode [256]rune
	for b := Token(0); b < 256; b++ {
		if _, ok := bytesUnicodeMap[uint8(b)]; !ok {
			bytesUnicodeMap[uint8(b)] = rune(256 + uct)
			unicodeBytes[rune(256+uct)] = uint8(b)
			uct += 1
		}
		bytesUnicode[b] = bytesUnicodeMap[uint8(b)]
	}

	// Go through the encoderMappings for possible byte runes.
	// Read Encoder mappings and also generate reverse mappings.
	bytesEncoder := make(map[byte]Token)
	encoderTokens := make(map[string]Token)
	if json.Unmarshal(*rsrcs["vocab.json"].Data, &encoderTokens) != nil {
		log.Fatal("Error unmarshalling `vocab.json`")
	}
	tokensEncoder := make(map[Token][]byte)
	for text, token := range encoderTokens {
		if strings.HasPrefix(text, "0x") && len(text) == 4 {
			// Convert the hexstring to a byte
			byteValue, err := strconv.ParseUint(text[2:], 16, 8)
			if err != nil {
				panic(err)
			}
			tokensEncoder[token] = []byte{byte(byteValue)}
			bytesEncoder[byte(byteValue)] = token
			delete(encoderTokens, text)
		} else {
			tokensEncoder[token] = []byte(text)
		}
	}
	bytesEncoderPtr := &bytesEncoder
	if len(bytesEncoder) == 0 {
		bytesEncoderPtr = nil
	}

	// Read merge table into BpeRanks
	bpeRanks := make(map[GPTPair]float64)
	if mergesTxt, ok := rsrcs["merges.txt"]; ok {
		scanner := bufio.NewScanner(bytes.NewBuffer(*mergesTxt.Data))
		idx := uint16(0)
		firstLine := true
		for scanner.Scan() {
			if firstLine == true {
				firstLine = false
				continue
			}
			left_right := strings.SplitN(scanner.Text(), " ", 2)
			bpeRanks[GPTPair{
				left_right[0],
				left_right[1]}] = float64(idx)
			idx += 1
		}
	} else if mergesJson, ok := rsrcs["merges.json"]; ok {
		var mergesTable [][]string
		err := json.Unmarshal(*mergesJson.Data, &mergesTable)
		if err != nil {
			panic(err)
		}
		// Iterate over the merges and add them to the BPE ranks
		for rank, merge := range mergesTable {
			bpeRanks[GPTPair{merge[0], merge[1]}] =
				float64(rank)
		}
	}
	// Build our TokenMerges
	tokenMerges := make(map[TokenPair]Token)
	for pair := range bpeRanks {
		tokenMerges[TokenPair{
			encoderTokens[pair.Left],
			encoderTokens[pair.Right]}] =
			encoderTokens[pair.Left+pair.Right]
	}

	// Handle special tokens. Special tokens are removed from the input before
	// tokenization, so we need to search for them before we tokenize.
	specialsRegexTokens := make([]string, 0)
	specials := make(map[string]Tokens, 0)
	specialsArr := make([]string, 0)

	if specialsTxt, ok := rsrcs["specials.txt"]; ok {
		specialsScanner := bufio.NewScanner(bytes.NewBuffer(*specialsTxt.Data))
		for specialsScanner.Scan() {
			specialToken := specialsScanner.Text()
			if specialToken == "" {
				continue
			}
			specials[specialToken] = Tokens{encoderTokens[specialToken]}
			specialsArr = append(specialsArr, specialToken)
			quotedToken := regexp.QuoteMeta(specialToken)
			specialsRegexTokens = append(specialsRegexTokens, quotedToken)
		}
	} else if specialsJson, ok := rsrcs["specials.json"]; ok {
		specialsData := make(map[string]string, 0)
		seenSpecials := make(map[string]bool, 0)
		if specialErr := json.Unmarshal(*specialsJson.Data,
			&specialsData); specialErr != nil {
			return nil, specialErr
		}
		for _, v := range specialsData {
			if _, seen := seenSpecials[v]; !seen {
				seenSpecials[v] = true
				specials[v] = Tokens{encoderTokens[v]}
				quotedToken := regexp.QuoteMeta(v)
				specialsRegexTokens = append(specialsRegexTokens, quotedToken)
			}
		}
	}
	specialsRegex := strings.Join(specialsRegexTokens, "|")

	// Now compile our regexes.
	specialsPat, err := regexp.Compile(specialsRegex)
	if err != nil {
		log.Fatalf(REGEX_ERROR, err)
	}

	var pat *regexp.Regexp
	if specialConfig.SplitRegex != nil {
		pat, err = regexp.Compile(*specialConfig.SplitRegex)
	} else {
		pat, err = regexp.Compile(SPLIT_REGEX)
	}
	if err != nil {
		log.Fatalf(REGEX_ERROR, err)
	}
	puncPat, err := regexp.Compile(PUNC_REGEX)
	if err != nil {
		log.Fatalf(REGEX_ERROR, err)
	}

	cache, _ := lru.NewARC(BPE_LRU_SZ)

	replacements := make(map[string]string, 0)
	if hfConfig != nil && hfConfig.Newlinemode != nil && *hfConfig.
		Newlinemode == "s" {
		replacements["\n"] = "</s>"
	}

	encoder := &GPTEncoder{
		encoderTokens,
		tokensEncoder,
		bpeRanks,
		tokenMerges,
		bytesEncoderPtr,
		unitrimArr,
		pat,
		puncPat,
		specialsPat,
		bytesUnicode,
		unicodeBytes,
		specials,
		specialsArr,
		nil,
		cache,
		puncRunes,
		normalizer,
		decodeExtra,
		encoderTokens[*hfConfig.BosTokenStr],
		encoderTokens[*hfConfig.EosTokenStr],
		encoderTokens[*hfConfig.PadTokenStr],
		specialConfig.EncloseEosBos,
		specialConfig.PrefixSpace,
		specialConfig.LowerCase,
		specialConfig.EndOfWord,
		replacements,
		RUNEBUF_SZ,
		WORDCHAN_SZ,
		0,
		0,
		0,
		BPE_LRU_SZ,
		4,
	}
	encoder.specialsTree = encoder.createRuneTree()
	return encoder, nil
}

// makeUnitrimArr creates a lookup table for unicode trimming
// it replaces the method of generating it in advanced (unitrim.json)
func makeUnitrimArr(encoderMap map[string]Token) []int {
	// get reverse mappings
	reverseEncoderMap := make(map[Token]string)
	for k, v := range encoderMap {
		reverseEncoderMap[v] = k
	}

	//make the maps needed
	preMapList := make([]int, 512)
	for i := 0; i < 512; i++ {
		preMapList[i] = -1
	}
	postMapList := make([]int, 512)

	// fill the maps with the unicode values
	postMapIdx, preMapIdx := 0, 0
	for i := uint8('!'); i < uint8('~')+1; i++ {
		preMapList[preMapIdx] = int(i)
		preMapIdx += 1
	}
	for i := uint8('¡'); i < uint8('¬')+1; i++ {
		preMapList[preMapIdx] = int(i)
		preMapIdx += 1
	}
	for i := uint16('®'); i < uint16('ÿ')+1; i++ {
		preMapList[preMapIdx] = int(i)
		preMapIdx += 1
	}
	copy(postMapList, preMapList)
	postMapIdx = preMapIdx

	for i := 0; i < 256; i++ {
		// if the value is not in the remapping list
		if !intInSlice(i, preMapList) {
			preMapList[preMapIdx] = i
			preMapIdx += 1
		}
	}
	preMapIdx = 0

	for i := 0; postMapIdx+i < 256; i++ {
		// continue incrementing until end of list
		if postMapList[postMapIdx+i] == -1 {
			postMapList[postMapIdx+i] = preMapList[postMapIdx+i]
			preMapList[postMapIdx+i] = preMapIdx + 256
			preMapIdx += 1
		}

	}

	// map int to rune
	charMap := make(map[int]rune)
	for i := 0; i < len(postMapList); i++ {
		charMap[postMapList[i]] = rune(postMapList[i])
	}

	// create the decode map
	DecodeMap := make(map[rune]int)
	for i := 0; i < 256; i++ {
		value := postMapList[i]
		key := rune(preMapList[i])
		DecodeMap[key] = value
	}

	// create the ordered array of encodings
	orderedArrayEncodings := make([]string, len(reverseEncoderMap))
	for k, v := range reverseEncoderMap {
		orderedArrayEncodings[k] = v
	}

	// create the need array (returned by this function)
	needArray := make([]int, 0)

	// for each encoding, get the need
	for i := 0; i < len(orderedArrayEncodings); i++ {
		need := 0
		min_need := 0

		// apply mapping
		v := DecodeMapping(reverseEncoderMap, DecodeMap, i)
		// get binary representation of each character
		// and match it to the need
		for _, c := range v {
			if (c & 0b10000000) == 0 {
				need = 0
			} else if (c & 0b11000000) == 0b10000000 {
				need -= 1
			} else if (c & 0b11100000) == 0b11000000 {
				need = 1
			} else if (c & 0b11110000) == 0b11100000 {
				need = 2
			} else if (c & 0b11111000) == 0b11110000 {
				need = 3
			}
			if need < 0 {
				min_need = need
			}

			if need == 0 {
				need = min_need
			}
		}
		needArray = append(needArray, need)
	}

	return needArray
}

// helper for makeUnitrimArr
func DecodeMapping(
	encodings map[Token]string,
	mappings map[rune]int,
	input int,
) string {
	// get original encoding
	encoding := encodings[Token(input)]

	// decode encoding by character
	output_str := ""
	for _, c := range encoding {
		// get the mapping for the character
		mapped := mappings[c]
		output_str += fmt.Sprintf("%c", mapped)
	}
	return output_str
}

// helper for makeUnitrimArr
func intInSlice(a int, list []int) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

// insertAt inserts v into s at index i and returns the new slice.
func insertAt(data []BGERank, i int, v BGERank) []BGERank {
	if i == len(data) {
		// Insert at end is the easy case.
		return append(data, v)
	}

	// Make space for the inserted element by shifting
	// values at the insertion index up one index. The call
	// to append does not allocate memory when cap(data) is
	// greater than len(data).
	data = append(data[:i+1], data[i:]...)

	// Insert the new element.
	data[i] = v

	// Return the updated slice.
	return data
}

// insertSortedNoDups inserts v, a BGERank, into data and returns the new slice.
// If v is already in data, it is not inserted again. It ensures that the slice
// is sorted and has no duplicates.
func insertSortedNoDups(data BGERanks, v BGERank) BGERanks {
	i := sort.Search(len(data), func(i int) bool {
		return data[i].rank >= v.rank
	})
	if i < len(data) && data[i] == v {
		return data
	}
	return insertAt(data, i, v)
}

func getPairs(word []string) []GPTPair {
	pairsSet := make(map[GPTPair]bool, len(word))
	pairs := make([]GPTPair, len(word))
	begin := 1
	prev := word[0]
	ct := 0
	for idx := begin; idx < len(word); idx++ {
		present := word[idx]
		pair := GPTPair{prev, present}
		if _, ok := pairsSet[pair]; !ok {
			pairs[len(pairsSet)] = pair
			ct++
		}
		pairsSet[pair] = true
		prev = present
	}
	return pairs[0:ct]
}

// getRankedPairs
// Accepts a slice of strings and returns a slice of BGERanks, sorted by
// their rank.
func (encoder *GPTEncoder) getRankedPairs(word []string) BGERanks {
	rankedPairs := make(BGERanks, 0, len(word))
	begin := 1
	prev := word[0]
	for idx := begin; idx < len(word); idx++ {
		present := word[idx]
		pair := GPTPair{prev, present}
		bpe, ok := encoder.BpeRanks[pair]
		if !ok {
			bpe = math.Inf(1)
		}
		rankedPairs = insertSortedNoDups(rankedPairs,
			BGERank{bpe, pair})
		prev = present
	}
	return rankedPairs
}

// rankPairs
// Accepts a slice of GPTPair and returns a slice of BGERanks, sorted by
// their rank.
func (encoder *GPTEncoder) rankPairs(pairs []GPTPair) BGERanks {
	rankedPairs := make(BGERanks, 0)
	for idx := range pairs {
		bpe, ok := encoder.BpeRanks[pairs[idx]]
		if !ok {
			bpe = math.Inf(1)
		}
		rankedPairs = insertSortedNoDups(rankedPairs,
			BGERank{bpe, pairs[idx]})
	}
	sort.Sort(rankedPairs)
	return rankedPairs
}

// minPair
// Accepts a slice of GPTPair and returns the pair with the lowest BPE rank.
func (encoder *GPTEncoder) minPair(pairs []GPTPair) (retPair GPTPair) {
	rankedPairs := encoder.rankPairs(pairs)
	if len(rankedPairs) > 0 {
		retPair = rankedPairs[0].bigram
	}
	return retPair
}

// pos finds the index of the first occurrence of seek in word past index i.
func pos(word []string, seek string, i int) int {
	for j, v := range word[i:] {
		if seek == v {
			return j + i
		}
	}
	return -1
}

// findAllStringIndex returns a set of indexes of all occurrences of substr in
// string.
func findAllStringIndex(text string, substr string) [][]int {
	var indexes [][]int
	for i := 0; i < len(text); {
		j := strings.Index(text[i:], substr)
		if j < 0 {
			break
		}
		indexes = append(indexes, []int{i + j, i + j + len(substr)})
		i += j + len(substr)
	}
	return indexes
}

// findAllStringsIndexes returns a set of indexes of all occurrences of strings,
// which are substrings of text removing all overlaps.
func findAllStringsIndexes(text string, strings []string) [][]int {
	var indexes [][]int
	for _, substr := range strings {
		indexes = append(indexes, findAllStringIndex(text, substr)...)
	}
	return indexes
}

// ToBPE
// Given pre-split text, perform bigram ranking and merges, and returns Tokens
func (encoder *GPTEncoder) ToBPE(text string) Tokens {
	if lookup, ok := encoder.Cache.Get(text); ok {
		encoder.LruHits++
		return lookup.(Tokens)
	} else {
		encoder.LruMisses++
	}
	word := strings.Split(text, "")
	word[len(word)-1] = word[len(word)-1] + encoder.endOfWord
	rankedPairs := encoder.getRankedPairs(word)
	if len(rankedPairs) == 0 {
		var tokens Tokens
		if token, ok := encoder.Encoder[word[0]]; ok {
			tokens = Tokens{token}
		} else if encoder.BytesEncoder != nil {
			tokens = make(Tokens, 0)
			rune_bytes := []byte(word[0])
			// Then encode each byte as a token.
			for _, b := range rune_bytes {
				tokens = append(tokens, (*encoder.BytesEncoder)[b])
			}
		} else {
			tokens = Tokens{encoder.Encoder[word[0]]}
		}
		encoder.Cache.Add(text, tokens)
		return tokens
	}
	for {
		bigram := rankedPairs[0].bigram
		if _, ok := encoder.BpeRanks[bigram]; !ok {
			break
		}
		first := bigram.Left
		second := bigram.Right
		newWord := make([]string, 0, len(word))
		for i := 0; i < len(word); {
			j := pos(word, first, i)
			if j == -1 {
				newWord = append(newWord, word[i:]...)
				break
			}
			newWord = append(newWord, word[i:j]...)
			i = j
			if word[i] == first && i < len(word)-1 && word[i+1] == second {
				newWord = append(newWord, first+second)
				i += 2
			} else {
				newWord = append(newWord, word[i])
				i += 1
			}
		}
		word = newWord
		if len(word) == 1 {
			break
		} else {
			rankedPairs = encoder.getRankedPairs(word)
		}
	}
	if len(word) > 0 {
		idx := len(word) - 1
		word[idx] = word[idx]
	}
	tokens := make(Tokens, 0)
	for _, token := range word {
		if lookup, ok := encoder.Encoder[token]; ok {
			tokens = append(tokens, lookup)
		} else if encoder.BytesEncoder != nil {
			// If we can't find the token in the encoder, we'll
			// encode it in byte-level BPE. First convert the rune
			// into 8-bit bytes.
			rune_bytes := []byte(token)
			// Then encode each byte as a token.
			for _, b := range rune_bytes {
				tokens = append(tokens, (*encoder.BytesEncoder)[b])
			}
		}
	}
	encoder.Cache.Add(text, tokens)
	return tokens
}

func (encoder *GPTEncoder) getSpecials() map[int][][]rune {
	lenMap := make(map[int][][]rune)
	for k := range encoder.Specials {
		keyLen := len(k)
		keyRunes := []rune(k)
		if entry, ok := lenMap[keyLen]; ok {
			lenMap[keyLen] = append(entry, keyRunes)
		} else {
			lenMap[keyLen] = [][]rune{keyRunes}
		}
	}
	return lenMap
}

func (encoder *GPTEncoder) splitWords(text string,
	specialToken bool, specialsNode *RuneNode) []*string {
	// Some things such as KoboldAI have a 'replacement' rule, where
	// they replace tokens such as `\n` with `</s>` for Fairseq
	// handling.
	for replaced, replacement := range encoder.replacements {
		text = strings.ReplaceAll(text, replaced, replacement)
	}
	text = encoder.Normalizer.Replace(text)

	idxes := encoder.pattern.FindAllStringIndex(text, -1)
	words := make([]*string, 0, len(idxes)+1)
	for idx := range idxes {
		word := text[idxes[idx][0]:idxes[idx][1]]
		if encoder.lowerCase {
			word = strings.ToLower(word)
		}

		if !encoder.prefixSpace {
			word = strings.TrimSpace(word)
		}

		if len(word) > 0 {
			words = append(words, &word)
		}
	}

	// Finally, if we have a special token, we cap it off.
	if specialToken {
		runeString := string(specialsNode.runes)
		words = append(words, &runeString)
	}
	return words
}

type NextRuneFunc func() (rune, int, error)
type WordCallback func(*string)

func (encoder *GPTEncoder) splitOntoChan(text string, ch chan *string,
	specialToken bool, specialsNode *RuneNode, wg *sync.WaitGroup) {
	defer close(ch)
	words := encoder.splitWords(text, specialToken, specialsNode)
	for _, word := range words {
		ch <- word
	}
	wg.Done()
}

func (encoder *GPTEncoder) synchronousSplitterThread(
	line string, specialToken bool, specialsNode *RuneNode,
	wg *sync.WaitGroup) chan *string {
	retCh := make(chan *string, 16)
	go encoder.splitOntoChan(line, retCh, specialToken, specialsNode, wg)
	return retCh
}

func (encoder *GPTEncoder) consumeSplitQueue(
	queue chan chan *string,
	cb WordCallback,
	wg *sync.WaitGroup) {
	for {
		select {
		case ch, ok := <-queue:
			if !ok {
				wg.Done()
				return
			}
			for word := range ch {
				cb(word)
			}
		}
	}
}

func (encoder *GPTEncoder) splitWordsOntoChan(
	nextRuneFunc NextRuneFunc,
	wordsChan chan *string,
) {
	specialsRuneRoot := encoder.specialsTree
	runeAccumulator := make([]rune, 0, encoder.runeBufSz)
	specialToken := false
	specialsNode := specialsRuneRoot
	for {
		// Let's collect runes until we reach the end of our IO stream, or
		// hit a newline.
		for {
			r, size, err := nextRuneFunc()
			if size == 0 || err != nil {
				break
			}

			runeAccumulator = append(runeAccumulator, r)
			specialsNode, specialToken = specialsRuneRoot.evaluate(
				specialsNode, r)
			if specialToken || r == '\n' {
				break
			}
		}

		// If we have no runes, then we've hit an error, or reached the end
		// of our IO stream.
		if len(runeAccumulator) == 0 {
			close(wordsChan)
			break
		}

		// If we've discovered a special token, then we need to split the
		// runeAccumulator before the special token.
		var line string
		if specialToken {
			line = string(runeAccumulator[:len(runeAccumulator)-len(
				specialsNode.runes)])
		} else {
			line = string(runeAccumulator)
		}
		runeAccumulator = runeAccumulator[:0]

		// We split all words before the special token in question, and
		// accumulate them.
		for _, word := range encoder.splitWords(line, specialToken,
			specialsNode) {
			wordsChan <- word
		}

		// Reset our special tokens state.
		specialsNode = specialsRuneRoot
		specialToken = false
	}
}

func (encoder *GPTEncoder) makeWordSplitter(
	nextRuneFunc NextRuneFunc,
	wordCallback WordCallback,
	completeCallback func()) func() {
	workQueue := make(chan chan *string, encoder.SplitterThreads)
	wg := sync.WaitGroup{}
	wg.Add(1)
	go encoder.consumeSplitQueue(workQueue, wordCallback, &wg)

	return func() {
		specialsRuneRoot := encoder.specialsTree
		runeAccumulator := make([]rune, 0, encoder.runeBufSz)
		specialToken := false
		specialsCandidates := make([]*RuneNode, 0, 16)
		candidateNode := specialsRuneRoot
		for {
			// Let's collect runes until we reach the end of our IO stream, or
			// hit a newline.
			for {
				r, size, err := nextRuneFunc()
				if size == 0 || err != nil {
					break
				}

				runeAccumulator = append(runeAccumulator, r)

				if r == '\n' {
					break
				}

				// Iterate over existing candidates, and see if we can
				// find a match.
				for idx, candidate := range specialsCandidates {
					candidate, specialToken = candidate.
						evaluate(candidate, r)
					specialsCandidates[idx] = candidate
					if specialToken {
						candidateNode = candidate
						break
					}
				}
				// Clean out any candidates that are no longer valid.
				for idx := 0; idx < len(specialsCandidates); idx++ {
					if idx >= len(specialsCandidates) {
						break
					}
					if specialsCandidates[idx] == nil {
						specialsCandidates = append(
							specialsCandidates[:idx],
							specialsCandidates[idx+1:]...)
						idx--
					}
				}

				if specialToken {
					break
				}

				candidateNode, specialToken = specialsRuneRoot.evaluate(
					specialsRuneRoot, r)
				if specialToken {
					break
				}
				if candidateNode != nil {
					specialsCandidates = append(specialsCandidates,
						candidateNode)
				}
			}

			// If we have no runes, then we've hit an error, or reached the end
			// of our IO stream.
			if len(runeAccumulator) == 0 {
				wordCallback(nil)
				break
			}

			// If we've discovered a special token, then we need to split the
			// runeAccumulator before the special token.
			var line string
			if specialToken {
				line = string(runeAccumulator[:len(runeAccumulator)-len(
					candidateNode.runes)])
			} else {
				line = string(runeAccumulator)
			}
			runeAccumulator = runeAccumulator[:0]

			// We split all words before the special token in question, and
			// accumulate them.
			wg.Add(1)
			workQueue <- encoder.synchronousSplitterThread(line, specialToken,
				candidateNode, &wg)

			// Reset our special tokens state.
			candidateNode = specialsRuneRoot
			specialToken = false
			specialsCandidates = specialsCandidates[:0]
		}
		close(workQueue)
		wg.Wait()
		completeCallback()
	}
}

// WordSplitter
// Returns an iterator function that reads from an io.RuneReader and splits
// the input into words. Each invocation of the iterator function returns
// one word or nil if there are no more words.
func (encoder *GPTEncoder) WordSplitter(reader io.RuneReader) func() *string {
	wordsAccumulator := make(chan string, encoder.wordChanSz)
	wordSplitter := encoder.makeWordSplitter(
		func() (rune, int, error) {
			return reader.ReadRune()
		},
		func(word *string) {
			if word != nil {
				wordsAccumulator <- *word
			}
		},
		func() {
			close(wordsAccumulator)
		})
	go wordSplitter()

	return func() *string {
		word, more := <-wordsAccumulator
		if more {
			return &word
		} else {
			return nil
		}
	}
}

// SplitWords splits a string into words according to BPE Encoder rules.
func (encoder *GPTEncoder) SplitWords(text *string) *[]string {
	words := make([]string, 0)
	nextWord := encoder.WordSplitter(strings.NewReader(*text))
	for {
		word := nextWord()
		if word == nil {
			break
		}
		words = append(words, *word)
	}
	return &words
}

func (encoder *GPTEncoder) toUnicode(text *string) string {
	if encoder.BytesEncoder != nil {
		runes := []rune(*text)
		return string(runes)
	}
	textBytes := []byte(*text)
	outArr := make([]rune, len(*text))
	for idx := range textBytes {
		outArr[idx] = encoder.byteToRune[textBytes[idx]]
	}
	return string(outArr)
}

func (encoder *GPTEncoder) encodeTokens(tokens *[]string) (encoded Tokens) {
	encoded = make(Tokens, len(*tokens))
	for idx := range *tokens {
		encoded[idx] = encoder.Encoder[(*tokens)[idx]]
	}
	return encoded
}

// StreamingEncode is a streaming Encoder. It takes an io.RuneReader and
// returns an iterator function that will return Tokens on each call.
func (encoder *GPTEncoder) StreamingEncode(reader io.RuneReader) func(int) *Tokens {
	nextWord := encoder.WordSplitter(reader)

	accumulator := make(Tokens, 0, 16384)
	eosReturned := false
	if encoder.encloseEosBos {
		accumulator = append(accumulator, encoder.BosToken)
	}
	return func(desiredTokens int) *Tokens {
		for {
			// If we have enough tokens, then we return them, and reset the
			// accumulator.
			if len(accumulator) > desiredTokens+1 {
				chunk := accumulator[:desiredTokens]
				accumulator = accumulator[desiredTokens:]
				return &chunk
			}
			// Fetch the next word from the WordSplitter.
			word := nextWord()
			// If we have no word, then we're done.
			if word == nil {
				if encoder.encloseEosBos && !eosReturned {
					accumulator = append(accumulator, encoder.EosToken)
					eosReturned = true
				}
				// If we have any tokens left, then we return them.
				if len(accumulator) > 0 {
					chunk := accumulator
					accumulator = accumulator[:0]
					return &chunk
				} else {
					return nil
				}
			}
			// Otherwise, we add the word to the accumulator. We have to handle
			// the special tokens here, since they're not in the vocab.
			var encodedTokens Tokens
			specialToken, isSpecial := encoder.Specials[*word]
			if isSpecial {
				decodedSpecial := string(encoder.Decoder[specialToken[0]])
				encodedTokens = Tokens{encoder.Encoder[decodedSpecial]}
			} else {
				fragment := encoder.toUnicode(word)
				encodedTokens = encoder.ToBPE(fragment)
			}
			accumulator = append(accumulator, encodedTokens...)
			if len(accumulator)-len(encodedTokens) > 0 {
				idx := len(accumulator) - len(encodedTokens) - 1
				for {
					pair := TokenPair{accumulator[idx],
						accumulator[idx+1]}
					if merged, ok := encoder.TokenMerges[pair]; ok {
						before := accumulator[:idx]
						var after Tokens
						if idx+2 < len(accumulator) {
							after = accumulator[idx+2:]
						}
						accumulator = append(before, merged)
						accumulator = append(accumulator, after...)
						if idx > 0 {
							idx -= 1
						}
					} else {
						idx += 1
					}
					if idx >= len(accumulator)-1 {
						break
					}
				}
			}
		}
	}
}

func (encoder *GPTEncoder) EncodeReader(reader io.RuneReader) *Tokens {
	encoded := make(Tokens, 0, 4096)
	nextTokens := encoder.StreamingEncode(reader)
	for {
		tokens := nextTokens(4096)
		if tokens == nil {
			break
		}
		encoded = append(encoded, *tokens...)
	}
	return &encoded
}

// EncodeBuffer takes a byte array and encodes it into Tokens in another
// byte array.
func (encoder *GPTEncoder) EncodeBuffer(buffer *[]byte) *[]byte {
	runeReader := bytes.NewReader(*buffer)
	nextTokens := encoder.StreamingEncode(runeReader)
	buf := bytes.NewBuffer(make([]byte, 0, 4096))
	for {
		tokens := nextTokens(2048)
		if tokens == nil {
			break
		}
		binary.Write(buf, binary.LittleEndian, tokens)
	}
	bufBytes := buf.Bytes()
	return &bufBytes
}

// Encode encodes a string into a sequence of tokens.
func (encoder *GPTEncoder) Encode(text *string) *Tokens {
	runeReader := strings.NewReader(*text)

	return encoder.EncodeReader(runeReader)
}

// Get
// Looks up text in the Encoder, and returns the Token representation of it. If
// the text is not found, then nil is returned.
func (encoder *GPTEncoder) Get(text string) *Token {
	if token, ok := encoder.Encoder[text]; !ok {
		return nil
	} else {
		return &token
	}
}

// Decode Tokens back into a string, handling unicode.
func (encoder *GPTEncoder) Decode(encoded *Tokens) (text string) {
	// Check if we have an end of word token defined.
	convertEndOfWord := false
	if encoder.endOfWord != "" {
		convertEndOfWord = true
	}
	// Accumulate tokens until it is unicode complete.
	tokensAcc := make(Tokens, 0)
	runesAcc := make([]rune, 0)

	for _, token := range *encoded {
		tokensAcc = append(tokensAcc, token)
		if encoder.TokensReady(&tokensAcc) {
			bs := make([]byte, 0, 32)
			for _, safeToken := range tokensAcc {
				if v, ok := encoder.Decoder[safeToken]; ok {
					bs = append(bs, v...)
				}
			}
			// Convert our bytearray to string, interpreting as UTF-8 and then
			// to 32-bit runes.
			// If we don't have a BytesEncoder, then we are using GPT BPE's
			// byte encoding for Unicode.
			var runes = []rune(string(bs))
			var fragment string
			if encoder.BytesEncoder == nil {
				decoded := make([]byte, len(runes))
				// Convert our runes into 8-bit bytes using a 256-slot table.
				for runeIdx := range runes {
					decoded[runeIdx] = encoder.runeToByte[runes[runeIdx]]
				}
				fragment = string(decoded)
				runes = []rune(fragment)
			} else {
				fragment = string(bs)
				runes = []rune(fragment)
			}
			// Decode our final token representation into a Unicode string.
			if convertEndOfWord {
				if strings.HasSuffix(fragment, encoder.endOfWord) {
					runes = runes[:len(runes)-len(encoder.endOfWord)]
					if len(runes) == 1 && runes[0] == '\'' {
					} else {
						runes = append(runes, ' ')
					}
				}
				if len(runes) == 1 &&
					unicode.IsNumber(runes[0]) {
					runes = append(runes, ' ')
				}
				// If we have a punctuation character, and the previous
				// character is a space, then we remove the space.
				// This is to handle cases like " ,".
				if len(runesAcc) > 1 && runeIsIn(runes[0],
					encoder.PuncRunes) && unicode.IsSpace(runesAcc[len(
					runesAcc)-1]) {
					runesAcc = runesAcc[:len(runesAcc)-1]
				}
			}
			runesAcc = append(runesAcc, runes...)
			tokensAcc = tokensAcc[:0]
		}
	}

	return string(runesAcc)
}

// DecodeBuffer
// Decode Tokens from a byte array into a string.
func (encoder *GPTEncoder) DecodeBuffer(encoded *[]byte) (text string) {
	// First convert our bytearray into a uint16 `Token` array.
	tokens := TokensFromBin(encoded)
	// Decode our tokens into a string.
	return encoder.Decode(tokens)
}

// TokensReady
// Determine if the sequence of Tokens given is ready to be serialized
// to string, based on if the sequence will produce valid Unicode runes.
func (encoder *GPTEncoder) TokensReady(tokens *Tokens) bool {
	if encoder.BytesEncoder != nil {
		return true
	}
	good := 0
	need := 0
	for tokenIdx := range *tokens {
		tok := (*tokens)[tokenIdx]
		var req int
		if int(tok) >= len(encoder.unitrim) {
			// Don't error out on tokens that we don't know about.
			req = 0
		} else {
			req = encoder.unitrim[(*tokens)[tokenIdx]]
		}

		if !(need+req < 0) {
			need += req
		}
		if req == 0 {
			// reset need to 0 to avoid being stuck when we have invalid
			// unicode being generated.
			need = 0
		}
		if need == 0 {
			good = tokenIdx + 1
		}
	}
	return good == len(*tokens)
}

// TrimTokens
// Trims the given Tokens to tokens that produce valid unicode.
func (encoder *GPTEncoder) TrimTokens(tokens *Tokens) (trimmed *Tokens) {
	trimmed = tokens
	for {
		if len(*trimmed) == 0 {
			return trimmed
		}
		if encoder.TokensReady(trimmed) {
			return trimmed
		} else {
			newTrimmed := (*trimmed)[0 : len(*trimmed)-1]
			trimmed = &newTrimmed
		}
	}
}

var GPT2Encoder = NewGPT2Encoder()
var PileEncoder = NewPileEncoder()
var CLIPEncoder = NewCLIPEncoder()
var NerdstashEncoder = NewNerdstashEncoder()
var blankString = ""
var _ = GPT2Encoder.Encode(&blankString)
var _ = PileEncoder.Encode(&blankString)
var _ = CLIPEncoder.Encode(&blankString)
var _ = NerdstashEncoder.Encode(&blankString)

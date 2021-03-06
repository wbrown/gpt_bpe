package gpt_bpe

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	lru "github.com/hashicorp/golang-lru"
	"github.com/wbrown/gpt_bpe/resources"
	"io"
	"log"
	"math"
	"regexp"
	"sort"
	"strings"
)

const BPE_LRU_SZ = 8192
const RUNEBUF_SZ = 16384
const WORDCHAN_SZ = 4096

type Token uint16
type Tokens []Token

type GPTEncoder struct {
	encoder      map[string]Token
	decoder      map[Token][]byte
	bpe_ranks    map[GPTPair]float64
	unitrim      []int
	pattern      *regexp.Regexp
	puncPat      *regexp.Regexp
	specialsPat  *regexp.Regexp
	byteToRune   [256]rune
	runeToByte   map[rune]byte
	specials     map[string]Tokens
	specialsTree *RuneNode
	cache        *lru.Cache
	EosToken     Token
	PadToken     Token
	replacements map[string]string
	runeBufSz    int
	wordChanSz   int
}

type GPTPair struct {
	left  string
	right string
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

type RuneNode struct {
	rune      rune
	runes     []rune
	terminal  bool
	childs    map[rune]*RuneNode
	childsArr []*RuneNode
}

func (encoder *GPTEncoder) createRuneTree() *RuneNode {
	runeTree := &RuneNode{
		runes:  []rune{},
		childs: make(map[rune]*RuneNode, 0),
	}

	for k := range encoder.specials {
		keyRunes := []rune(k)
		keyLen := len(keyRunes)
		node := runeTree
		for i := 0; i < keyLen; i++ {
			r := keyRunes[i]
			childNode, ok := node.childs[r]
			if !ok {
				node.childs[r] = &RuneNode{
					rune:      r,
					runes:     keyRunes[:i+1],
					terminal:  i == keyLen-1,
					childs:    make(map[rune]*RuneNode, 0),
					childsArr: make([]*RuneNode, 0),
				}
			} else if i == keyLen-1 {
				childNode.terminal = true
			}
			if len(node.childs) != len(node.childsArr) {
				node.childsArr = append(node.childsArr, node.childs[r])
			}
			node = node.childs[r]
		}
	}
	return runeTree
}

func (root *RuneNode) evaluate(node *RuneNode, r rune) (*RuneNode, bool) {
	for childIdx := range node.childsArr {
		child := node.childsArr[childIdx]
		if child.rune == r {
			return child, child.terminal
		}
	}
	if node != root {
		return root.evaluate(root, r)
	} else {
		return root, false
	}
}

// NewEncoder
// Returns a GPTEncoder with the tokenizer data loaded for that vocabulary
// id.
func NewEncoder(vocabId string) (*GPTEncoder, error) {
	hfConfig, resourcesPtr, vocabErr := resources.ResolveVocabId(vocabId)
	if vocabErr != nil {
		return nil, vocabErr
	}
	rsrcs := *resourcesPtr

	if hfConfig != nil && hfConfig.ModelId != nil {
		vocabId = *hfConfig.ModelId
	}

	// Read token unicode trimming definitions
	unitrimArr := make([]int, 0)
	if json.Unmarshal(*rsrcs["unitrim.json"].Data, &unitrimArr) != nil {
		log.Fatal("Error unmarshalling `unitrim.json`")
	}

	// Read encoder mappings and also generate reverse mappings.
	encoderTokens := make(map[string]Token)
	if json.Unmarshal(*rsrcs["vocab.json"].Data, &encoderTokens) != nil {
		log.Fatal("Error unmarshalling `vocab.json`")
	}
	tokensEncoder := make(map[Token][]byte)
	for text, token := range encoderTokens {
		tokensEncoder[token] = []byte(text)
	}
	// Read vocabulary into bpe_ranks
	bpeRanks := make(map[GPTPair]float64)
	scanner := bufio.NewScanner(bytes.NewBuffer(*rsrcs["merges.txt"].Data))
	idx := uint16(0)
	firstLine := true
	for scanner.Scan() {
		if firstLine == true {
			firstLine = false
			continue
		}
		left_right := strings.SplitN(scanner.Text(), " ", 2)
		bpeRanks[GPTPair{left_right[0], left_right[1]}] = float64(idx)
		idx += 1
	}

	// Handle special tokens. Special tokens are removed from the input before
	// tokenization, so we need to search for them before we tokenize.
	specialsRegexTokens := make([]string, 0)
	specials := make(map[string]Tokens, 0)

	if specialsTxt, ok := rsrcs["specials.txt"]; ok {
		specialsScanner := bufio.NewScanner(bytes.NewBuffer(*specialsTxt.Data))
		for specialsScanner.Scan() {
			specialToken := specialsScanner.Text()
			if specialToken == "" {
				continue
			}
			specials[specialToken] = Tokens{encoderTokens[specialToken]}
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
	pat, err := regexp.Compile(SPLIT_REGEX)
	if err != nil {
		log.Fatalf(REGEX_ERROR, err)
	}
	puncPat, err := regexp.Compile(PUNC_REGEX)
	if err != nil {
		log.Fatalf(REGEX_ERROR, err)
	}
	// Build the bytes to unicode tables.
	bytesUnicodeMap := make(map[byte]rune)
	unicodeBytes := make(map[rune]byte)
	for b := uint8('!'); b < uint8('~')+1; b++ {
		bytesUnicodeMap[b] = rune(b)
		unicodeBytes[rune(b)] = b
	}
	for b := uint8('??'); b < uint8('??')+1; b++ {
		bytesUnicodeMap[b] = rune(b)
		unicodeBytes[rune(b)] = b
	}
	for b := uint16('??'); b < uint16('??')+1; b++ {
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

	cache, _ := lru.New(BPE_LRU_SZ)

	replacements := make(map[string]string, 0)
	if hfConfig != nil && hfConfig.Newlinemode != nil && *hfConfig.
		Newlinemode == "s" {
		replacements["\n"] = "</s>"
	}

	encoder := &GPTEncoder{
		encoderTokens,
		tokensEncoder,
		bpeRanks,
		unitrimArr,
		pat,
		puncPat,
		specialsPat,
		bytesUnicode,
		unicodeBytes,
		specials,
		nil,
		cache,
		encoderTokens[*hfConfig.EosTokenStr],
		encoderTokens[*hfConfig.PadTokenStr],
		replacements,
		RUNEBUF_SZ,
		WORDCHAN_SZ,
	}
	encoder.specialsTree = encoder.createRuneTree()
	return encoder, nil
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
		bpe, ok := encoder.bpe_ranks[pair]
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
		bpe, ok := encoder.bpe_ranks[pairs[idx]]
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

// toBPE
// Given pre-split text, return a list of BPE tokens as strings.
func (encoder *GPTEncoder) toBPE(text string) []string {
	if lookup, ok := encoder.cache.Get(text); ok {
		return lookup.([]string)
	}
	word := strings.Split(text, "")
	rankedPairs := encoder.getRankedPairs(word)
	if len(rankedPairs) == 0 {
		return []string{text}
	}
	for {
		bigram := rankedPairs[0].bigram
		if _, ok := encoder.bpe_ranks[bigram]; !ok {
			break
		}
		first := bigram.left
		second := bigram.right
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
	encoder.cache.Add(text, word)
	return word
}

func (encoder *GPTEncoder) getSpecials() map[int][][]rune {
	lenMap := make(map[int][][]rune)
	for k := range encoder.specials {
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

// WordSplitter
// Returns an iterator function that reads from an io.RuneReader and splits
// the input into words. Each invocation of the iterator function returns
// one word or nil if there are no more words.
func (encoder *GPTEncoder) WordSplitter(reader io.RuneReader) func() *string {
	wordsAccumulator := make(chan string, encoder.wordChanSz)

	wordSplitter := func() {
		specialsRuneRoot := encoder.specialsTree
		runeAccumulator := make([]rune, 0, encoder.runeBufSz)
		specialToken := false
		specialsNode := specialsRuneRoot
		for {
			// Let's collect runes until we reach the end of our IO stream, or
			// hit a newline.
			for {
				r, size, err := reader.ReadRune()
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
				close(wordsAccumulator)
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

			// Some things such as KoboldAI have a 'replacement' rule, where
			// they replace tokens such as `\n` with `</s>` for Fairseq
			// handling.
			for replaced, replacement := range encoder.replacements {
				line = strings.ReplaceAll(line, replaced, replacement)
			}

			// We split all words before the special token in question, and
			// accumulate them.
			idxes := encoder.pattern.FindAllStringIndex(line, -1)
			for idx := range idxes {
				word := line[idxes[idx][0]:idxes[idx][1]]
				wordsAccumulator <- word
			}

			// Finally, if we have a special token, we cap it off.
			if specialToken {
				wordsAccumulator <- string(specialsNode.runes)
			}

			// Reset our special tokens state.
			specialsNode = specialsRuneRoot
			specialToken = false
		}
	}

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

// SplitWords splits a string into words according to BPE encoder rules.
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
	textBytes := []byte(*text)
	outArr := make([]rune, len(*text))
	for idx := range textBytes {
		outArr[idx] = encoder.byteToRune[textBytes[idx]]
	}
	return string(outArr)
}

func (encoder *GPTEncoder) encodeTokens(tokens *[]string) (encoded Tokens) {
	for idx := range *tokens {
		encoded = append(encoded, encoder.encoder[(*tokens)[idx]])
	}
	return encoded
}

// StreamingEncode is a streaming encoder. It takes an io.RuneReader and
// returns an iterator function that will return Tokens on each call.
func (encoder *GPTEncoder) StreamingEncode(reader io.RuneReader) func(int) *Tokens {
	nextWord := encoder.WordSplitter(reader)
	accumulator := make(Tokens, 0, 16384)
	return func(desiredTokens int) *Tokens {
		for {
			// If we have enough tokens, then we return them, and reset the
			// accumulator.
			if len(accumulator) > desiredTokens {
				chunk := accumulator[:desiredTokens]
				accumulator = accumulator[desiredTokens:]
				return &chunk
			}
			// Fetch the next word from the WordSplitter.
			word := nextWord()
			// If we have no word, then we're done.
			if word == nil {
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
			specialToken, isSpecial := encoder.specials[*word]
			if isSpecial {
				decodedSpecial := string(encoder.decoder[specialToken[0]])
				encodedTokens = Tokens{encoder.encoder[decodedSpecial]}
			} else {
				fragment := encoder.toUnicode(word)
				token := encoder.toBPE(fragment)
				encodedTokens = encoder.encodeTokens(&token)
			}
			accumulator = append(accumulator, encodedTokens...)
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
// Looks up text in the encoder, and returns the Token representation of it. If
// the text is not found, then nil is returned.
func (encoder *GPTEncoder) Get(text string) *Token {
	if token, ok := encoder.encoder[text]; !ok {
		return nil
	} else {
		return &token
	}
}

// Decode Tokens back into a string, handling unicode.
func (encoder *GPTEncoder) Decode(encoded *Tokens) (text string) {
	// First convert our `Token` tokens into an 8-bit byte array.
	bs := make([]byte, 0, len(*encoded))
	for idx := range *encoded {
		if v, ok := encoder.decoder[(*encoded)[idx]]; ok {
			for bIdx := range v {
				bs = append(bs, v[bIdx])
			}
		}
	}
	// Convert our bytearray to string, interpreting as UTF-8 and then to
	// 32-bit runes.
	runes := []rune(string(bs))
	decoded := make([]byte, len(runes))
	// Convert our runes into 8-bit bytes using a 256-slot lookup table.
	for runeIdx := range runes {
		decoded[runeIdx] = encoder.runeToByte[runes[runeIdx]]
	}
	// Decode our final representation into an Unicode string.
	text = string(decoded)
	return text
}

// TokensReady
// Determine if the sequence of Tokens given is ready to be serialized
// to string, based on if the sequence will produce valid Unicode runes.
func (encoder *GPTEncoder) TokensReady(tokens *Tokens) bool {
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
var blankString = ""
var _ = GPT2Encoder.Encode(&blankString)
var _ = PileEncoder.Encode(&blankString)

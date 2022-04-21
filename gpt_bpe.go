package gpt_bpe

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	lru "github.com/hashicorp/golang-lru"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
)

const BPE_LRU_SZ = 8192

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
	cache        *lru.Cache
	EosToken     Token
	PadToken     Token
	replacements map[string]string
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
	encoder, _ := NewEncoder("gpt2")
	return *encoder
}

func NewPileEncoder() GPTEncoder {
	encoder, _ := NewEncoder("pile")
	return *encoder
}

type HFConfig struct {
	ModelType      *string `json:"model_type,omitempty"`
	EosTokenId     *Token  `json:"eos_token_id,omitempty"`
	BosTokenId     *Token  `json:"bos_token_id,omitempty"`
	PadTokenId     *Token  `json:"pad_token_id,omitempty"`
	EosTokenStr    *string `json:"eos_token,omitempty"`
	PadTokenStr    *string `json:"pad_token,omitempty"`
	VocabSize      *uint16 `json:"vocab_size,omitempty"`
	Newlinemode    *string `json:"newlinemode,omitempty"`
	TokenizerClass *string `json:"tokenizer_class"`
}

func FetchHuggingFace(vocabId string, file string) (respBytes []byte,
	err error) {
	resp, remoteErr := http.Get("https://huggingface." +
		"co/" + vocabId + "/raw/main/" + file)
	if remoteErr != nil {
		return nil, remoteErr
	}
	defer resp.Body.Close()
	respBytes, respErr := io.ReadAll(resp.Body)
	if respErr != nil {
		return nil, respErr
	} else {
		if mkdirErr := os.MkdirAll(vocabId, 0755); mkdirErr != nil {
			return respBytes, mkdirErr
		}
		os.WriteFile(vocabId+"/"+file, respBytes, 0755)
		return respBytes, nil
	}
}

func ResolveHF(vocabId string) (exists bool, err error, config *HFConfig) {
	configJson, configErr := FetchHuggingFace(vocabId, "config.json")
	if configErr != nil {
		return false, configErr, nil
	}

	var hfConfig HFConfig
	configErr = json.Unmarshal(configJson, &hfConfig)
	if configErr != nil {
		return true, configErr, nil
	}
	vocabJson, vocabErr := FetchHuggingFace(vocabId, "vocab.json")
	if vocabErr != nil {
		return true, vocabErr, nil
	}
	mergesTxt, mergesErr := FetchHuggingFace(vocabId, "merges.txt")
	if mergesErr != nil {
		return true, mergesErr, nil
	}
	specialJson, specialErr := FetchHuggingFace(vocabId,
		"special_tokens_map.json")

	var writeErr error

	if writeErr = os.WriteFile(vocabId+"/vocab.json", vocabJson,
		0755); writeErr != nil {
		return true, writeErr, nil
	}
	if writeErr = os.WriteFile(vocabId+"/merges.txt", mergesTxt,
		0755); writeErr != nil {
		return true, writeErr, nil
	}

	specialTokens := make(map[string]interface{}, 0)
	if specialErr == nil {
		if specialErr = json.Unmarshal(specialJson,
			&specialTokens); specialErr != nil {
			return true, specialErr, nil
		}
	}

	seenSpecials := make(map[string]bool)
	specialTokenStrings := make([]string, 0)
	specialsFile, specialFileErr := os.OpenFile(vocabId+"/specials.txt",
		os.O_TRUNC|os.O_WRONLY|os.O_CREATE, 0755)
	if specialFileErr != nil {
		return true, specialFileErr, nil
	}
	defer specialsFile.Close()
	for k, v := range specialTokens {
		var specialToken string
		switch t := v.(type) {
		case string:
			specialToken = t
		case map[string]interface{}:
			mv := t["content"]
			switch mvt := mv.(type) {
			case string:
				specialToken = mvt
			default:
				log.Fatal(fmt.Sprintf("Unknown format for `special_tokens_map."+
					"json`: %v", t))
			}
		default:
			log.Fatal(fmt.Sprintf("Unknown format for `special_tokens_map."+
				"json`: %v", t))
		}
		if !seenSpecials[specialToken] {
			specialTokenStrings = append(specialTokenStrings, specialToken)
			seenSpecials[specialToken] = true
		}
		switch k {
		case "eos_token":
			hfConfig.EosTokenStr = &specialToken
		case "pad_token":
			hfConfig.PadTokenStr = &specialToken
		}
	}
	if len(specialTokenStrings) > 0 {
		_, writeErr = specialsFile.WriteString(strings.Join(
			specialTokenStrings, "\n"))
		if writeErr != nil {
			return true, writeErr, nil
		}
	}
	defaultTkn := "<|endoftext|>"
	if hfConfig.EosTokenStr == nil {
		hfConfig.EosTokenStr = &defaultTkn
	}
	if hfConfig.PadTokenStr == nil {
		hfConfig.PadTokenStr = &defaultTkn
	}

	hfConfigJson, _ := json.Marshal(hfConfig)
	if writeErr = os.WriteFile(vocabId+"/config.json", hfConfigJson,
		0755); writeErr != nil {
		return true, writeErr, nil
	}
	return true, nil, &hfConfig
}

func ResolveVocabId(vocabId string) (bool, *HFConfig, error) {
	if _, vocabErr := f.ReadDir("resources/" + vocabId); vocabErr == nil {
		return true, nil, nil
	}
	if _, localErr := os.ReadDir(vocabId); localErr == nil {
		configJson, configErr := os.ReadFile(vocabId + "/config.json")
		if configErr != nil {
			return false, nil, configErr
		}
		var hfConfig HFConfig
		configErr = json.Unmarshal(configJson, &hfConfig)
		if configErr != nil {
			return false, nil, configErr
		}
		return false, &hfConfig, nil
	}
	if hfExists, hfErr, hfConfig := ResolveHF(vocabId); hfErr != nil {
		if hfExists {
			return false, nil, hfErr
		} else {
			return false, nil, errors.New(fmt.Sprintf(
				"Unknown tokenizer vocabulary '%s", vocabId))
		}
	} else {
		return false, hfConfig, nil
	}
}

func NewEncoder(vocabId string) (*GPTEncoder, error) {
	isInternalVocab, hfConfig, vocabErr := ResolveVocabId(vocabId)
	if vocabErr != nil {
		return nil, vocabErr
	}

	var eosTokenStr, padTokenStr string
	var unitrimFile, encoderFile, ranksFile, specialsFile []byte
	if isInternalVocab {
		eosTokenStr = "<|endoftext|>"
		padTokenStr = "<|endoftext|>"
		unitrimFile, _ = f.ReadFile(
			"resources/" + vocabId + "/unitrim.json")
		encoderFile, _ = f.ReadFile(
			"resources/" + vocabId + "/encoder.json")
		ranksFile, _ = f.ReadFile(
			"resources/" + vocabId + "/vocab.bpe")
		specialsFile, _ = f.ReadFile(
			"resources/" + vocabId + "/specials.txt")
	} else {
		eosTokenStr = *hfConfig.EosTokenStr
		padTokenStr = *hfConfig.PadTokenStr
		unitrimFile, _ = f.ReadFile(
			"resources/gpt2/unitrim.json")
		encoderFile, _ = os.ReadFile(vocabId + "/vocab.json")
		ranksFile, _ = os.ReadFile(vocabId + "/merges.txt")
		specialsFile, _ = os.ReadFile(vocabId + "/specials.txt")
	}

	// Read token unicode trimming definitions
	unitrimArr := make([]int, 0)
	if json.Unmarshal(unitrimFile, &unitrimArr) != nil {
		log.Fatal("Error unmarshalling `unitrim.json`")
	}

	// Read encoder mappings and also generate reverse mappings.
	encoderTokens := make(map[string]Token)
	if json.Unmarshal(encoderFile, &encoderTokens) != nil {
		log.Fatal("Error unmarshalling encoder")
	}
	tokensEncoder := make(map[Token][]byte)
	for text, token := range encoderTokens {
		tokensEncoder[token] = []byte(text)
	}
	// Read vocabulary into bpe_ranks
	bpeRanks := make(map[GPTPair]float64)
	scanner := bufio.NewScanner(bytes.NewBuffer(ranksFile))
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

	specialsScanner := bufio.NewScanner(bytes.NewBuffer(specialsFile))
	for specialsScanner.Scan() {
		specialToken := specialsScanner.Text()
		if specialToken == "" {
			continue
		}
		specials[specialToken] = Tokens{encoderTokens[specialToken]}
		quotedToken := regexp.QuoteMeta(specialToken)
		specialsRegexTokens = append(specialsRegexTokens, quotedToken)
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

	cache, _ := lru.New(BPE_LRU_SZ)

	replacements := make(map[string]string, 0)
	if hfConfig != nil && hfConfig.Newlinemode != nil && *hfConfig.
		Newlinemode == "s" {
		replacements["\n"] = "</s>"
	}

	return &GPTEncoder{
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
		cache,
		encoderTokens[eosTokenStr],
		encoderTokens[padTokenStr],
		replacements,
	}, nil
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
	// greater ​than len(data).
	data = append(data[:i+1], data[i:]...)

	// Insert the new element.
	data[i] = v

	// Return the updated slice.
	return data
}

func insertSortedNoDups(data BGERanks, v BGERank) BGERanks {
	i := sort.Search(len(data), func(i int) bool { return data[i].rank >= v.rank })
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

func (encoder *GPTEncoder) minPair(pairs []GPTPair) (retPair GPTPair) {
	rankedPairs := encoder.rankPairs(pairs)
	if len(rankedPairs) > 0 {
		retPair = rankedPairs[0].bigram
	}
	return retPair
}

func (encoder *GPTEncoder) toUnicode(text *string) string {
	textBytes := []byte(*text)
	outArr := make([]rune, len(*text))
	for idx := range textBytes {
		outArr[idx] = encoder.byteToRune[textBytes[idx]]
	}
	return string(outArr)
}

func pos(word []string, seek string, i int) int {
	for j, v := range word[i:] {
		if seek == v {
			return j + i
		}
	}
	return -1
}

func (encoder *GPTEncoder) encodeTokens(tokens *[]string) (encoded Tokens) {
	for idx := range *tokens {
		encoded = append(encoded, encoder.encoder[(*tokens)[idx]])
	}
	return encoded
}

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

func (encoder *GPTEncoder) SplitWords(text *string) *[]string {
	splitLines := strings.SplitAfter(*text, "\n")
	words := make([]string, 0, len(*text)/3)
	for lineIdx := 0; lineIdx < len(splitLines); lineIdx++ {
		line := splitLines[lineIdx]
		for lineIdx < len(splitLines)-1 {
			if splitLines[lineIdx+1] == "\n" {
				line = line + "\n"
				lineIdx += 1
			} else {
				break
			}
		}
		for replaced, replacement := range encoder.replacements {
			line = strings.ReplaceAll(line, replaced, replacement)
		}
		specialIdxes := encoder.specialsPat.FindAllStringIndex(line, -1)
		beginIdx := 0
		var specialEnd int
		for specialIdx := range specialIdxes {
			specialBegin := specialIdxes[specialIdx][0]
			specialEnd = specialIdxes[specialIdx][1]
			specialSplit := line[specialBegin:specialEnd]
			beforeSpecial := line[beginIdx:specialBegin]
			idxes := encoder.pattern.FindAllStringIndex(beforeSpecial, -1)
			for idx := range idxes {
				words = append(words,
					beforeSpecial[idxes[idx][0]:idxes[idx][1]])
			}
			words = append(words, specialSplit)
			beginIdx = specialEnd
		}
		if specialEnd < len(line) {
			post := line[specialEnd:]
			idxes := encoder.pattern.FindAllStringIndex(post, -1)
			for idx := range idxes {
				words = append(words, post[idxes[idx][0]:idxes[idx][1]])
			}
		}
	}
	return &words
}

func (encoder *GPTEncoder) Encode(text *string) *Tokens {
	words := encoder.SplitWords(text)
	encoded := make(Tokens, 0)
	for idx := range *words {
		var encodedTokens Tokens
		if specialToken, isSpecial := encoder.specials[(*words)[idx]]; isSpecial {
			encodedTokens = Tokens{encoder.encoder[string(encoder.decoder[specialToken[0]])]}
		} else {
			fragment := encoder.toUnicode(&(*words)[idx])
			token := encoder.toBPE(fragment)
			encodedTokens = encoder.encodeTokens(&token)
		}
		encoded = append(encoded, encodedTokens...)
	}
	return &encoded
}

func (encoder *GPTEncoder) Get(text string) *Token {
	if token, ok := encoder.encoder[text]; !ok {
		return nil
	} else {
		return &token
	}
}

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
		if int(tok) > len(encoder.unitrim) {
			req = 0
		} else {
			req = encoder.unitrim[(*tokens)[tokenIdx]]
		}
		if !(need+req < 0) {
			need += req
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

package resources

import (
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/vikesh-raj/go-sentencepiece-encoder/sentencepiece"
	"google.golang.org/protobuf/proto"
)

var escaper *strings.Replacer

type DuplicateEntry struct {
	OldIdx int
	NewIdx int
	Repr   string
}

type GPTPair struct {
	Left  string
	Right string
}

type VocabEntry struct {
	TokenId *uint32
	Token   *string
	ByteId  *uint32
	Byte    *string
}

type SentencePieceVocab struct {
	TokenToPiece []VocabEntry
	PieceToToken map[string]VocabEntry
}

func EscapeString(
	s string,
) (escaped string) {
	if escaper == nil {
		escaper = strings.NewReplacer(
			"\"", "\\\"",
			"\\", "\\\\",
			"\n", "\\n",
			"\r", "\\r",
			"\b", "\\b",
			"\t", "\\t")
	}
	escaped = escaper.Replace(s)
	asRunes := []rune(escaped)
	if len(asRunes) == 1 && (unicode.IsControl(asRunes[0]) ||
		!unicode.IsPrint(asRunes[0])) {
		escaped = fmt.Sprintf("\\u%04x", asRunes[0])
	}
	return escaped
}

func UnescapeString(
	s string,
	verbose bool,
) (unescaped string) {
	if strings.HasPrefix(s, "\\u") {
		// Unescape unicode
		code, _ := hex.DecodeString(s[2:6])
		unescaped = string(code)
		if verbose {
			print(fmt.Sprintf("Unescaped unicode: %v -> %v", s, unescaped))
		}
	} else {
		unescaped = s
	}
	return unescaped
}

func GenerateVocab(
	model *sentencepiece.ModelProto,
) (
	vocab *SentencePieceVocab,
	duplicates *[]DuplicateEntry,
	specials *[]string,
) {
	vocab = &SentencePieceVocab{
		TokenToPiece: make([]VocabEntry, len(model.GetPieces())+1),
		PieceToToken: make(map[string]VocabEntry),
	}
	specials = &[]string{}
	duplicateEntries := make([]DuplicateEntry, 0)
	duplicates = &duplicateEntries
	spaceReplacer := strings.NewReplacer(
		"▁", " ")
	// Build the vocab
	for pieceIdx, piece := range model.GetPieces() {
		repr := piece.GetPiece()
		pieceIsByte := piece.GetType() ==
			sentencepiece.ModelProto_SentencePiece_BYTE
		pieceIsControl := piece.GetType() ==
			sentencepiece.ModelProto_SentencePiece_CONTROL
		pieceIsUser := piece.GetType() ==
			sentencepiece.ModelProto_SentencePiece_USER_DEFINED
		if pieceIsByte {
			hexRepr := piece.GetPiece()[3:5]
			encodedRepr, _ := hex.DecodeString(hexRepr)
			repr = string(encodedRepr)
		} else {
			repr = spaceReplacer.Replace(repr)
			if pieceIsControl || pieceIsUser {
				*specials = append(*specials, repr)
			}
		}
		if dupeEntry, ok := vocab.PieceToToken[repr]; ok {
			var dupeIdx uint32
			if dupeEntry.TokenId != nil {
				dupeIdx = *dupeEntry.TokenId
			} else {
				dupeIdx = *dupeEntry.ByteId
			}
			if pieceIsByte {
				byteToken := uint32(pieceIdx)
				dupeEntry.Byte = &repr
				dupeEntry.ByteId = &byteToken
			} else {
				tokenToken := uint32(pieceIdx)
				dupeEntry.Token = &repr
				dupeEntry.TokenId = &tokenToken
			}
			vocab.PieceToToken[repr] = dupeEntry
			vocab.TokenToPiece[dupeIdx] = dupeEntry
			vocab.TokenToPiece[uint32(pieceIdx)] = dupeEntry
			print(fmt.Sprintf("Duplicate piece: old (%v): %v, dupe ("+
				"%v): %v\n",
				dupeIdx, model.GetPieces()[dupeIdx], pieceIdx, piece))
			*duplicates = append(*duplicates, DuplicateEntry{
				OldIdx: int(dupeIdx),
				NewIdx: pieceIdx,
				Repr:   repr,
			})
		} else {
			if pieceIsByte {
				byteToken := uint32(pieceIdx)
				vocab.PieceToToken[repr] = VocabEntry{
					Byte:   &repr,
					ByteId: &byteToken,
				}
			} else {
				tokenToken := uint32(pieceIdx)
				vocab.PieceToToken[repr] = VocabEntry{
					Token:   &repr,
					TokenId: &tokenToken,
				}
			}
			vocab.TokenToPiece[pieceIdx] = vocab.PieceToToken[repr]
		}
	}
	return vocab, duplicates, specials
}

func GenerateMergeTable(
	vocab *SentencePieceVocab,
	verbose bool,
) map[GPTPair]uint32 {
	// Build the merge table
	mergeTable := make(map[GPTPair]uint32, 0)

	// Loop over the model and print out the pieces
	currPair := GPTPair{"", ""}
	for _, token := range vocab.TokenToPiece {
		if token.Token == nil || *token.Token == "" || len(*token.Token) < 2 {
			continue
		}
		for splitIdx := 1; splitIdx < len(*token.Token); splitIdx++ {
			currPair.Left = (*token.Token)[:splitIdx]
			currPair.Right = (*token.Token)[splitIdx:]
			// Check if both pieces exist in the vocab
			leftTokenEntry, leftOk := vocab.PieceToToken[currPair.Left]
			rightTokenEntry, rightOk := vocab.PieceToToken[currPair.Right]
			if !leftOk || !rightOk {
				continue
			}
			if _, ok := mergeTable[currPair]; !ok {
				mergedToken := fmt.Sprintf("%v%v",
					currPair.Left,
					currPair.Right)

				if tokenEntry, ok := vocab.PieceToToken[mergedToken]; ok {
					leftTokenId := leftTokenEntry.TokenId
					rightTokenId := rightTokenEntry.TokenId
					tokenId := *tokenEntry.TokenId
					if verbose {
						print(fmt.Sprintf("%v (%v) %v (%v) -> %v (%v)\n",
							currPair.Left, leftTokenId,
							currPair.Right, rightTokenId,
							mergedToken, tokenId))
					}
					mergeTable[currPair] = tokenId
				}
			}
		}
	}
	return mergeTable
}

// Our struct for the merge array
type MergeEntry struct {
	Left        string `json:"left"`
	LeftToken   uint32 `json:"-"`
	Right       string `json:"right"`
	RightToken  uint32 `json:"-"`
	Merged      string `json:"-"`
	MergedToken uint32 `json:"-"`
}

func GenerateMergeEntries(
	vocab *SentencePieceVocab,
	mergeTable map[GPTPair]uint32,
) []MergeEntry {
	// Turn the merge table into an array of entries
	mergeEntries := make([]MergeEntry, 0)
	for pair := range mergeTable {
		mergedToken := fmt.Sprintf("%v%v", pair.Left, pair.Right)
		// Skip single rune tokens
		if len([]rune(mergedToken)) == 1 {
			continue
		}
		mergeEntries = append(mergeEntries,
			MergeEntry{pair.Left,
				*vocab.PieceToToken[pair.Left].TokenId,
				pair.Right,
				*vocab.PieceToToken[pair.Right].TokenId,
				mergedToken,
				*vocab.PieceToToken[mergedToken].TokenId})
	}
	// Sort the merge array by token id
	sort.Slice(mergeEntries, func(i, j int) bool {
		return mergeEntries[i].MergedToken < mergeEntries[j].MergedToken
	})
	return mergeEntries
}

func WriteDuplicates(
	name string,
	duplicates *[]DuplicateEntry,
) {
	duplicatesFile, err := os.Create(fmt.Sprintf("%s.json", name))
	if err != nil {
		panic(err)
	}
	duplicatesFile.WriteString("[\n")
	for idx, dupe := range *duplicates {
		escaped := EscapeString(dupe.Repr)
		duplicatesFile.WriteString(fmt.Sprintf("  {\"old_id\": %v, "+
			"\"new_id\": %v, \"repr\": \"%v\"}",
			dupe.OldIdx, dupe.NewIdx, escaped))
		if idx != len(*duplicates)-1 {
			duplicatesFile.WriteString(",\n")
		} else {
			duplicatesFile.WriteString("\n")
		}
	}
	duplicatesFile.WriteString("]\n")
}

func WriteMergeFiles(
	name string,
	mergeEntries []MergeEntry,
	verbose bool,
) {
	mergesFile, err := os.Create(fmt.Sprintf("%s.json", name))
	if err != nil {
		panic(err)
	}

	if verbose {
		mergesFile.WriteString("[\n")
	} else {
		mergesFile.WriteString("[")
	}

	// Write the merge table to a text file and json file
	for idx, pair := range mergeEntries {
		leftRepr := EscapeString(pair.Left)
		rightRepr := EscapeString(pair.Right)
		mergedRepr := EscapeString(pair.Merged)

		if idx != 0 && verbose {
			mergesFile.WriteString(",\n  ")
		} else if idx != 0 {
			mergesFile.WriteString(",")
		}

		if verbose {
			mergesFile.WriteString(fmt.Sprintf(
				"{\"left\": \"%v\", \", left_token\": %v, "+
					"\"right\": \"%v\", \"right_token\": %v, "+
					"\"merged\": \"%v\", \"merged_token\": %v}",
				leftRepr, pair.LeftToken,
				rightRepr, pair.RightToken,
				mergedRepr, pair.MergedToken))
		} else {
			mergesFile.WriteString(fmt.Sprintf(
				"[\"%v\",\"%v\"]",
				leftRepr, rightRepr))
		}
	}
	if verbose {
		mergesFile.WriteString("]")
	} else {
		mergesFile.WriteString("\n]\n")
	}
	mergesFile.Close()
}

func WriteVocabFile(
	name string,
	vocab *SentencePieceVocab,
	verbose bool,
) {
	// Serialize vocab to a JSON file
	vocabFile, _ := os.Create(fmt.Sprintf("%s.json", name))
	vocabSize := len(vocab.TokenToPiece)

	var entryPrefix string
	if verbose {
		entryPrefix = " "
		vocabFile.WriteString("{\n")
	} else {
		entryPrefix = ""
		vocabFile.WriteString("{")
	}

	for tokenId := 0; tokenId < vocabSize; tokenId++ {
		tokenEntry := vocab.TokenToPiece[tokenId]
		var repr string
		if tokenEntry.TokenId != nil &&
			*tokenEntry.TokenId == uint32(tokenId) {
			repr = EscapeString(*tokenEntry.Token)
		} else if tokenEntry.Byte != nil {
			// Convert our repr string to a byte
			reprByte := []byte(*tokenEntry.Byte)
			// Convert the byte to a hexstring
			repr = fmt.Sprintf("0x%02x", reprByte)
		}
		if tokenId != 0 && verbose {
			vocabFile.WriteString(",\n")
		} else if tokenId != 0 {
			vocabFile.WriteString(",")
		}

		vocabFile.WriteString(fmt.Sprintf("%s\"%v\":%s%d",
			entryPrefix, repr, entryPrefix, tokenId))
	}
	if verbose {
		vocabFile.WriteString("\n}\n")
	} else {
		vocabFile.WriteString("}")
	}
	vocabFile.Close()
}

func WriteSpecials(
	name string,
	specials *[]string,
) {
	// Sort the specials by length
	sort.Slice(*specials, func(i, j int) bool {
		return len((*specials)[i]) < len((*specials)[j])
	})

	specialsFile, err := os.Create(fmt.Sprintf("%s.txt", name))
	if err != nil {
		panic(err)
	}
	for idx, special := range *specials {
		if idx != 0 {
			specialsFile.WriteString("\n")
		}
		specialsFile.WriteString(fmt.Sprintf("%s", special))
	}
	specialsFile.Close()
}

func ConvertSentencepieceFiles(modelPath string, verbose bool) {
	bytes, err := ioutil.ReadFile(modelPath)
	if err != nil {
		print(fmt.Errorf("Unable to read file err %v", err))
	}
	var model sentencepiece.ModelProto
	err = proto.Unmarshal(bytes, &model)
	if err != nil {
		print(fmt.Errorf("Unable to unmarshal proto err %v", err))
	}
	//get parent of modelPath
	outputPath := path.Dir(modelPath)

	vocab, duplicates, specials := GenerateVocab(&model)
	WriteVocabFile(path.Join(outputPath, "vocab"), vocab, verbose)
	WriteSpecials(path.Join(outputPath, "specials"), specials)
	WriteDuplicates(path.Join(outputPath, "duplicates"), duplicates)
	mergeTable := GenerateMergeTable(vocab, verbose)
	mergeEntries := GenerateMergeEntries(vocab, mergeTable)
	WriteMergeFiles(path.Join(outputPath, "merges"), mergeEntries, verbose)
}

func main() {
	start := time.Now()
	ConvertSentencepieceFiles("./tokenizer.model", true)
	elapsed := time.Since(start)
	fmt.Printf("Conversion took %s\n", elapsed)
}

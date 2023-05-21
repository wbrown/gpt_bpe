package gpt_bpe

import (
	"io"
	"strings"
	"testing"
)

var sanitizeTable = map[string]string{
	"â‚¬": "€",
	"â€š": "‚",
	"Æ’":  "ƒ",
	"â€ž": "„",
	"â€¦": "…",
	"â€¡": "‡",
	"Ë†":  "ˆ",
	"â€°": "‰",
	"â€¹": "‹",
	"Å’":  "Œ",
	"Å½":  "Ž",
	"â€˜": "‘",
	"â€™": "’",
	"â€œ": "“",
	"â€¢": "•",
	"â€“": "–",
	"â€”": "—",
	"Ëœ":  "˜",
	"â„¢": "™",
	"Å¡":  "š",
	"â€º": "›",
	"Å“":  "œ",
	"Å¾":  "ž",
	"Å¸":  "Ÿ",
	"Â¡":  "¡",
	"Â¢":  "¢",
	"Â£":  "£",
	"Â¤":  "¤",
	"Â¥":  "¥",
	"Â¦":  "¦",
	"Â§":  "§",
	"Â¨":  "¨",
	"Â©":  "©",
	"Âª":  "ª",
	"Â«":  "«",
	"Â®":  "®",
	"Â¯":  "¯",
	"Â°":  "°",
	"Â±":  "±",
	"Â²":  "²",
	"Â³":  "³",
	"Â´":  "´",
	"Âµ":  "µ",
	"Â¶":  "¶",
	"Â·":  "·",
	"Â¸":  "¸",
	"Â¹":  "¹",
	"Âº":  "º",
	"Â»":  "»",
	"Â¼":  "¼",
	"Â½":  "½",
	"Â¾":  "¾",
	"Â¿":  "¿",
	"Ã€":  "À",
	"Ã‚":  "Â",
	"Ãƒ":  "Ã",
	"Ã„":  "Ä",
	"Ã…":  "Å",
	"Ã†":  "Æ",
	"Ã‡":  "Ç",
	"Ãˆ":  "È",
	"Ã‰":  "É",
	"ÃŠ":  "Ê",
	"Ã‹":  "Ë",
	"ÃŒ":  "Ì",
	"ÃŽ":  "Î",
	"Ã‘":  "Ñ",
	"Ã’":  "Ò",
	"Ã“":  "Ó",
	"Ã”":  "Ô",
	"Ã•":  "Õ",
	"Ã–":  "Ö",
	"Ã—":  "×",
	"Ã˜":  "Ø",
	"Ã™":  "Ù",
	"Ãš":  "Ú",
	"Ã›":  "Û",
	"Ãœ":  "Ü",
	"Ãž":  "Þ",
	"ÃŸ":  "ß",
	"Ã¡":  "á",
	"Ã¢":  "â",
	"Ã£":  "ã",
	"Ã¤":  "ä",
	"Ã¥":  "å",
	"Ã¦":  "æ",
	"Ã§":  "ç",
	"Ã¨":  "è",
	"Ã©":  "é",
	"Ãª":  "ê",
	"Ã«":  "ë",
	"Ã¬":  "ì",
	"Ã®":  "î",
	"Ã¯":  "ï",
	"Ã°":  "ð",
	"Ã±":  "ñ",
	"Ã²":  "ò",
	"Ã³":  "ó",
	"Ã´":  "ô",
	"Ãµ":  "õ",
	"Ã¶":  "ö",
	"Ã·":  "÷",
	"Ã¸":  "ø",
	"Ã¹":  "ù",
	"Ãº":  "ú",
	"Ã»":  "û",
	"Ã¼":  "ü",
	"Ã½":  "ý",
	"Ã¾":  "þ",
	"Ã¿":  "ÿ",
}

var encodingSanitzer = map[string]string{}

func TestRuneNode_String(t *testing.T) {
	print(nerdstashV2Encoder.SpecialsTree.String())
}

func TestRuneMatch(t *testing.T) {
	s := "// TypeScript Version: 2.9"
	rr := io.RuneReader(strings.NewReader(s))
	nextWord := nerdstashV2Encoder.WordSplitter(rr)
	for {
		word := nextWord()
		if word == nil {
			break
		}
		t.Log(*word)
	}
}

func TestRuneReplacement(t *testing.T) {
	s := "Ã¹ TypeScriptÃ–"
	rr := io.RuneReader(strings.NewReader(s))
	nerdstashV2Encoder.SpecialsTree.InsertReplacementsIntoRuneTree(
		sanitizeTable)
	print(nerdstashV2Encoder.SpecialsTree.String())
	nextWord := nerdstashV2Encoder.WordSplitter(rr)
	for {
		word := nextWord()
		if word == nil {
			break
		}
		t.Log(*word)
	}
}

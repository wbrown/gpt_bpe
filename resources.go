package gpt_bpe

import "embed"

//go:embed resources/encoder.json
//go:embed resources/vocab.bpe
//go:embed resources/gpt-2-unitrim.json

var f embed.FS

var unitrim []int

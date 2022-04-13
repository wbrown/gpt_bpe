package gpt_bpe

import "embed"

//go:embed resources/gpt2/encoder.json
//go:embed resources/gpt2/vocab.bpe
//go:embed resources/gpt2/unitrim.json
//go:embed resources/gpt2/specials.txt
//go:embed resources/pile/encoder.json
//go:embed resources/pile/vocab.bpe
//go:embed resources/pile/unitrim.json
//go:embed resources/pile/specials.txt
var f embed.FS

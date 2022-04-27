package resources

import "embed"

//go:embed gpt2-tokenizer/encoder.json
//go:embed gpt2-tokenizer/vocab.bpe
//go:embed gpt2-tokenizer/unitrim.json
//go:embed gpt2-tokenizer/specials.txt
//go:embed pile-tokenizer/encoder.json
//go:embed pile-tokenizer/vocab.bpe
//go:embed pile-tokenizer/unitrim.json
//go:embed pile-tokenizer/specials.txt
var f embed.FS

package types

type Token uint32
type Tokens []Token
type TokenMap map[string]Token

const (
	TokenSize = 2
)

type GPTPair struct {
	Left  string
	Right string
}

type TokenPair struct {
	Left  Token
	Right Token
}

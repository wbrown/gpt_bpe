package gpt_bpe

import "errors"

func (encoder *GPTEncoder) TrimIncompleteSentence(
	tokens *Tokens,
) (*Tokens, error) {
	return nil, errors.New("TrimIncompleteSentence is not implemented")
}

func (encoder *GPTEncoder) TrimSentences(
	tokens *Tokens,
	direction TrimDirection,
	limit uint,
) (*Tokens, error) {
	return nil, errors.New("TrimSentences is not implemented")
}

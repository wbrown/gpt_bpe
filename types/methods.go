package types

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

func (tokens *Tokens) ToBin(useUint32 bool) (*[]byte, error) {
	if useUint32 {
		return tokens.ToBinUint32()
	} else {
		return tokens.ToBinUint16()
	}
}

func (tokens *Tokens) ToBinUint16() (*[]byte, error) {
	buf := bytes.NewBuffer(make([]byte, 0, len(*tokens)*2))
	for idx := range *tokens {
		bs := (*tokens)[idx]
		if bs > 65535 {
			return nil, fmt.Errorf("integer overflow: tried to write token ID %d as unsigned 16-bit", bs)
		}
		err := binary.Write(buf, binary.LittleEndian, uint16(bs))
		if err != nil {
			return nil, err
		}
	}
	byt := buf.Bytes()
	return &byt, nil
}

func (tokens *Tokens) ToBinUint32() (*[]byte, error) {
	buf := bytes.NewBuffer(make([]byte, 0, len(*tokens)*4))
	for idx := range *tokens {
		bs := (*tokens)[idx]
		err := binary.Write(buf, binary.LittleEndian, uint32(bs))
		if err != nil {
			return nil, err
		}
	}
	byt := buf.Bytes()
	return &byt, nil
}

func TokensFromBin(bin *[]byte) *Tokens {
	tokens := make(Tokens, 0, len(*bin)/2)
	buf := bytes.NewReader(*bin)
	for {
		var token uint16
		if err := binary.Read(buf, binary.LittleEndian, &token); err != nil {
			break
		}
		tokens = append(tokens, Token(token))
	}
	return &tokens
}

func TokensFromBin32(bin *[]byte) *Tokens {
	tokens := make(Tokens, 0, len(*bin)/4)
	buf := bytes.NewReader(*bin)
	for {
		var token Token
		if err := binary.Read(buf, binary.LittleEndian, &token); err != nil {
			break
		}
		tokens = append(tokens, token)
	}
	return &tokens
}

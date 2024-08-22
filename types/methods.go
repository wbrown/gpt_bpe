package types

import (
	"bytes"
	"encoding/binary"
)

func (tokens *Tokens) ToBin(useUint32 bool) *[]byte {
	if useUint32 {
		return tokens.ToBinUint32()
	} else {
		return tokens.ToBinUint16()
	}
}

func (tokens *Tokens) ToBinUint16() *[]byte {
	buf := bytes.NewBuffer(make([]byte, 0, len(*tokens)*TokenSize))
	for idx := range *tokens {
		bs := (*tokens)[idx]
		binary.Write(buf, binary.LittleEndian, uint16(bs))
	}
	byt := buf.Bytes()
	return &byt
}

func (tokens *Tokens) ToBinUint32() *[]byte {
	buf := bytes.NewBuffer(make([]byte, 0, len(*tokens)*TokenSize))
	for idx := range *tokens {
		bs := (*tokens)[idx]
		binary.Write(buf, binary.LittleEndian, uint32(bs))
	}
	byt := buf.Bytes()
	return &byt
}

func TokensFromBin(bin *[]byte) *Tokens {
	type tokenuint16 uint16
	tokens := make(Tokens, 0)
	buf := bytes.NewReader(*bin)
	for {
		var token tokenuint16
		if err := binary.Read(buf, binary.LittleEndian, &token); err != nil {
			break
		}
		tu32 := Token(uint16(token))
		tokens = append(tokens, tu32)
	}
	return &tokens
}

func TokensFromBin32(bin *[]byte) *Tokens {
	tokens := make(Tokens, 0)
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

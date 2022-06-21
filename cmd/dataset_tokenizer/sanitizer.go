package main

import (
	"bufio"
	"bytes"
	"io"
	"regexp"
	"strings"
)

type SanitizedRuneReader struct {
	bufSize         int
	lastRune        *rune
	whitespaceRegex *regexp.Regexp
	reader          *bufio.Reader
	currBuffer      **bytes.Buffer
	accumulator     *[]rune
	accumulatorIdx  *int
	moreBuffers     chan *bytes.Buffer
}

func (runeReader SanitizedRuneReader) nextBuffer() *bytes.Buffer {
	acc := runeReader.accumulator
	accIdx := runeReader.accumulatorIdx
	idx := *accIdx
	var text string
	for {
		if idx > runeReader.bufSize {
			text = string((*acc)[:runeReader.bufSize])
			(*acc)[0] = (*acc)[idx-1]
			*accIdx = 1
			break
		}
		r, size, _ := (*runeReader.reader).ReadRune()
		if size == 0 && idx == 0 {
			// No valid rune, and our accumulator is empty, so we're done.
			return nil
		} else if size == 0 {
			// No valid rune, and we have stuff in our accumulator, so let's
			// flush and finish up.
			text = string((*acc)[:*accIdx])
			*accIdx = 0
			break
		} else if r == '\r' {
			// Silently drop Windows `\r`
		} else if r == '\n' && *runeReader.lastRune == '\n' {
			// Drop additional newlines.
		} else if r == 'n' && *runeReader.lastRune == '\\' {
			// Replace escaped `\n` with `\n`.
			(*acc)[idx-1] = '\n'
		} else if r == ':' && *runeReader.lastRune == ' ' {
			// Strip colons with leading spaces.
			(*acc)[idx-1] = ':'
		} else if r == '\t' {
			// Replace tabs with single spaces.
			(*acc)[idx] = ' '
			idx++
		} else {
			// We have a valid rune, so let's insert it onto our accumulator.
			(*acc)[idx] = r
			idx++
		}
		*runeReader.lastRune = (*acc)[idx-1]
	}
	lines := strings.Split(text, "\n")
	for lineIdx := range lines {
		line := lines[lineIdx]
		line = runeReader.whitespaceRegex.ReplaceAllString(line, " ")
		line = strings.TrimSpace(line)
		lines[lineIdx] = line
	}
	text = strings.Join(lines, "\n")
	stringBuffer := bytes.NewBufferString(text)
	return stringBuffer
}

func (runeReader SanitizedRuneReader) ReadRune() (r rune, size int,
	err error) {
	if *runeReader.currBuffer == nil {
		return 0, 0, io.EOF
	} else if r, size, err = (*runeReader.currBuffer).ReadRune(); err != nil {
		if newBuffer, ok := <-runeReader.moreBuffers; !ok {
			return 0, 0, io.EOF
		} else {
			*runeReader.currBuffer = newBuffer
			return (*runeReader.currBuffer).ReadRune()
		}
	} else {
		return r, size, err
	}
}

func CreateTextSanitizer(handle io.Reader) SanitizedRuneReader {
	extraWhiteSpace := regexp.MustCompile("[[:space:]]+")
	scanner := bufio.NewReader(handle)
	accumulator := make([]rune, 32769, 32769)
	emptyBuffer := bytes.NewBufferString("")
	lastRune := rune(0)
	accumlatorIdx := 0
	sanitizer := SanitizedRuneReader{
		bufSize:         32768,
		lastRune:        &lastRune,
		whitespaceRegex: extraWhiteSpace,
		reader:          scanner,
		accumulator:     &accumulator,
		accumulatorIdx:  &accumlatorIdx,
		currBuffer:      &emptyBuffer,
		moreBuffers:     make(chan *bytes.Buffer, 1),
	}
	nextBuffer := sanitizer.nextBuffer()
	sanitizer.currBuffer = &nextBuffer
	go func() {
		for {
			if newBuffer := sanitizer.nextBuffer(); newBuffer == nil {
				close(sanitizer.moreBuffers)
				break
			} else {
				sanitizer.moreBuffers <- newBuffer
			}
		}
	}()
	return sanitizer
}

func SanitizeText(text string) string {
	reader := CreateTextSanitizer(bytes.NewBufferString(text))
	runes := make([]rune, 0)
	for {
		r, size, _ := reader.ReadRune()
		if size > 0 {
			runes = append(runes, r)
		} else {
			break
		}
	}
	return string(runes)
}

package vocab

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// Token is a single vocabulary token ID.
type Token int32

const (
	tokenTypeNormal    = uint32(0)
	tokenTypeSpecial   = uint32(1)
	tokenTypeTimestamp = uint32(3)
)

// SpecialIDs contains the IDs of all special tokens.
type SpecialIDs struct {
	SOT        Token
	EOT        Token
	Blank      Token
	NoSpeech   Token
	Translate  Token
	Transcribe Token
	Prev       Token
	SOLM       Token
	NotSOT     Token
	// Timestamp range
	TimestampBegin Token
	TimestampEnd   Token
	// Language tokens (index = language index, not language ID)
	Languages []Token
}

// Vocabulary maps between token strings and IDs and holds special-token IDs.
type Vocabulary struct {
	tokens      []string
	tokenToID   map[string]Token
	tokenTypes  []uint32
	special     SpecialIDs
	byteDecoder [256]byte // indexed by (rune - 256) for overflow runes; identity runes use direct cast
	byteEncoder [256]rune // byte → unicode char (bytes_to_unicode forward mapping)
}

var (
	// reTimestamp matches timestamp tokens like <|0.00|>, <|30.00|>
	reTimestamp = regexp.MustCompile(`^<\|\d+\.\d+\|>$`)
	// reLanguage matches 2–3 letter BCP-47 language tokens like <|en|>, <|yue|> (but not <|startoftranscript|>)
	reLanguage = regexp.MustCompile(`^<\|[a-z]{2,3}\|>$`)
	// reSplit is the GPT-2 BPE word-splitting pattern (lookahead omitted for RE2 compat)
	reSplit = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)
)

// buildBytesUnicode constructs the GPT-2 bytes_to_unicode forward and reverse mappings.
//
// Identity bytes: 33–126 (printable ASCII), 161–172 (¡–¬), 174–255 (®–ÿ) map to themselves.
// The remaining 68 bytes map to U+0100, U+0101, … in byte-value order.
func buildBytesUnicode() (enc [256]rune, dec [256]byte) {
	inSet := [256]bool{}
	for i := '!'; i <= '~'; i++ {
		enc[i] = i
		inSet[i] = true
	}
	for i := rune('¡'); i <= rune('¬'); i++ {
		enc[i] = i
		inSet[i] = true
	}
	for i := rune('®'); i <= rune('ÿ'); i++ {
		enc[i] = i
		inSet[i] = true
	}
	n := rune(0x100)
	for i := 0; i < 256; i++ {
		if !inSet[i] {
			enc[i] = n
			dec[n-0x100] = byte(i)
			n++
		}
	}
	return enc, dec
}

// runeToBytes maps a BPE rune back to its original byte.
func (v *Vocabulary) runeToByte(r rune) (byte, bool) {
	if r >= 33 && r < 256 {
		return byte(r), true
	}
	if r >= 0x100 && r < 0x100+256 {
		return v.byteDecoder[r-0x100], true
	}
	return 0, false
}

// New builds a Vocabulary from parallel token/type slices (loaded from GGUF).
func New(tokens []string, tokenTypes []uint32) (*Vocabulary, error) {
	if len(tokens) != len(tokenTypes) {
		return nil, fmt.Errorf("vocab: tokens length %d != tokenTypes length %d", len(tokens), len(tokenTypes))
	}

	enc, dec := buildBytesUnicode()

	v := &Vocabulary{
		tokens:      make([]string, len(tokens)),
		tokenToID:   make(map[string]Token, len(tokens)),
		tokenTypes:  make([]uint32, len(tokenTypes)),
		byteEncoder: enc,
		byteDecoder: dec,
	}
	copy(v.tokens, tokens)
	copy(v.tokenTypes, tokenTypes)

	for i, tok := range tokens {
		v.tokenToID[tok] = Token(i)
	}

	// Detect and assign special tokens by string.
	// blankToken is the BPE representation of a space byte (0x20 → U+0120 = Ġ).
	blankToken := string([]rune{rune(0x0120)}) // Ġ

	specialNames := map[string]*Token{
		"<|startoftranscript|>": &v.special.SOT,
		"<|endoftext|>":         &v.special.EOT,
		blankToken:              &v.special.Blank,
		"<|nospeech|>":          &v.special.NoSpeech,
		"<|translate|>":         &v.special.Translate,
		"<|transcribe|>":        &v.special.Transcribe,
		"<|startofprev|>":       &v.special.Prev,
		"<|startoflm|>":         &v.special.SOLM,
		"<|notimestamps|>":      &v.special.NotSOT,
	}

	// Initialise all named special tokens to -1 (not found).
	v.special.SOT = -1
	v.special.EOT = -1
	v.special.Blank = -1
	v.special.NoSpeech = -1
	v.special.Translate = -1
	v.special.Transcribe = -1
	v.special.Prev = -1
	v.special.SOLM = -1
	v.special.NotSOT = -1
	v.special.TimestampBegin = -1
	v.special.TimestampEnd = -1

	var languages []Token

	for i, tok := range tokens {
		id := Token(i)

		if ptr, ok := specialNames[tok]; ok {
			*ptr = id
			continue
		}

		if reTimestamp.MatchString(tok) {
			if v.special.TimestampBegin == -1 {
				v.special.TimestampBegin = id
			}
			v.special.TimestampEnd = id
			continue
		}

		if tokenTypes[i] == tokenTypeSpecial && reLanguage.MatchString(tok) {
			languages = append(languages, id)
		}
	}

	v.special.Languages = languages
	return v, nil
}

// Encode tokenises text into token IDs using byte-level BPE.
//
// Each byte of the UTF-8 input is mapped through bytes_to_unicode, the result is
// split on word boundaries, and each segment is looked up in the token map.
// If a segment is not found as a whole, individual characters are looked up.
func (v *Vocabulary) Encode(text string) []Token {
	if text == "" {
		return nil
	}

	// Map every UTF-8 byte through bytes_to_unicode.
	var bpeText strings.Builder
	bpeText.Grow(len(text))
	for _, b := range []byte(text) {
		bpeText.WriteRune(v.byteEncoder[b])
	}
	encoded := bpeText.String()

	segments := reSplit.FindAllString(encoded, -1)
	result := make([]Token, 0, len(segments))

	for _, seg := range segments {
		if id, ok := v.tokenToID[seg]; ok {
			result = append(result, id)
			continue
		}
		// Character-by-character fallback.
		for _, ch := range seg {
			if id, ok := v.tokenToID[string(ch)]; ok {
				result = append(result, id)
			}
		}
	}

	return result
}

// tokenToBytes reverses the bytes_to_unicode mapping for a single token string.
func (v *Vocabulary) tokenToBytes(t Token) []byte {
	if int(t) < 0 || int(t) >= len(v.tokens) {
		return nil
	}
	s := v.tokens[t]
	out := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := v.runeToByte(r); ok {
			out = append(out, b)
		}
	}
	return out
}

// Decode converts token IDs back to a UTF-8 string, skipping non-normal tokens.
func (v *Vocabulary) Decode(tokens []Token) string {
	var buf []byte
	for _, t := range tokens {
		if int(t) < 0 || int(t) >= len(v.tokenTypes) {
			continue
		}
		if v.tokenTypes[t] != tokenTypeNormal {
			continue
		}
		buf = append(buf, v.tokenToBytes(t)...)
	}
	return string(buf)
}

// DecodeToken returns the UTF-8 string for a single token (any type).
func (v *Vocabulary) DecodeToken(t Token) string {
	return string(v.tokenToBytes(t))
}

// Special returns the special token IDs.
func (v *Vocabulary) Special() SpecialIDs { return v.special }

// Size returns the vocabulary size.
func (v *Vocabulary) Size() int { return len(v.tokens) }

// IsTimestamp reports whether t is a timestamp token.
func (v *Vocabulary) IsTimestamp(t Token) bool {
	return t >= v.special.TimestampBegin && t <= v.special.TimestampEnd && v.special.TimestampBegin >= 0
}

// TimestampToMs converts a timestamp token to milliseconds.
// Returns 0 for non-timestamp tokens.
func (v *Vocabulary) TimestampToMs(t Token) int64 {
	if !v.IsTimestamp(t) {
		return 0
	}
	s := v.tokens[t] // e.g. "<|1.50|>"
	inner := s[2 : len(s)-2]
	secs, err := strconv.ParseFloat(inner, 64)
	if err != nil {
		return 0
	}
	return int64(math.Round(secs * 1000))
}

// LanguageID returns the token ID for the given BCP-47 language code (e.g. "en").
func (v *Vocabulary) LanguageID(lang string) (Token, bool) {
	id, ok := v.tokenToID["<|"+lang+"|>"]
	return id, ok
}

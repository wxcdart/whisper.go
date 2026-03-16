package vocab

// Token is a single vocabulary token ID.
type Token int32

// Vocabulary maps between token strings and IDs and holds special-token IDs.
type Vocabulary struct{ /* unexported */ }

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
	// Language tokens (index = language ID)
	Languages []Token
}

// New builds a Vocabulary from parallel token/type slices (loaded from GGUF).
func New(tokens []string, tokenTypes []uint32) (*Vocabulary, error) { panic("not implemented") }

// Encode tokenises text into token IDs (BPE).
func (v *Vocabulary) Encode(text string) []Token { panic("not implemented") }

// Decode converts token IDs back to a UTF-8 string.
func (v *Vocabulary) Decode(tokens []Token) string { panic("not implemented") }

// DecodeToken returns the string for a single token.
func (v *Vocabulary) DecodeToken(t Token) string { panic("not implemented") }

// Special returns the special token IDs.
func (v *Vocabulary) Special() SpecialIDs { panic("not implemented") }

// Size returns the vocabulary size (n_vocab).
func (v *Vocabulary) Size() int { panic("not implemented") }

// IsTimestamp reports whether t is a timestamp token.
func (v *Vocabulary) IsTimestamp(t Token) bool { panic("not implemented") }

// TimestampToMs converts a timestamp token to milliseconds.
func (v *Vocabulary) TimestampToMs(t Token) int64 { panic("not implemented") }

// LanguageID returns the token ID for the given BCP-47 language code.
func (v *Vocabulary) LanguageID(lang string) (Token, bool) { panic("not implemented") }

package vocab

import (
	"testing"
)

// buildTestVocab creates a small hand-crafted vocabulary suitable for unit tests.
func buildTestVocab(t *testing.T) *Vocabulary {
	t.Helper()

	blankToken := string([]rune{rune(0x0120)}) // Ġ — BPE representation of space

	tokens := []string{
		// 0–2: normal tokens
		"hi",       // 0
		"Hello",    // 1
		blankToken, // 2  (blank / space in BPE)
		// 3–11: named special tokens
		"<|startoftranscript|>", // 3
		"<|endoftext|>",         // 4
		"<|nospeech|>",          // 5
		"<|translate|>",         // 6
		"<|transcribe|>",        // 7
		"<|startofprev|>",       // 8
		"<|startoflm|>",         // 9
		"<|notimestamps|>",      // 10
		// 11–12: language tokens
		"<|en|>", // 11
		"<|fr|>", // 12
		// 13–15: timestamp tokens
		"<|0.00|>",  // 13
		"<|1.50|>",  // 14
		"<|30.00|>", // 15
	}
	types := []uint32{
		0, // hi
		0, // Hello
		0, // Ġ
		1, // startoftranscript
		1, // endoftext
		1, // nospeech
		1, // translate
		1, // transcribe
		1, // startofprev
		1, // startoflm
		1, // notimestamps
		1, // en
		1, // fr
		3, // 0.00
		3, // 1.50
		3, // 30.00
	}

	v, err := New(tokens, types)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return v
}

func TestNew(t *testing.T) {
	v := buildTestVocab(t)

	if v.Size() != 16 {
		t.Errorf("Size() = %d, want 16", v.Size())
	}

	sp := v.Special()

	checks := []struct {
		name string
		got  Token
		want Token
	}{
		{"SOT", sp.SOT, 3},
		{"EOT", sp.EOT, 4},
		{"Blank", sp.Blank, 2},
		{"NoSpeech", sp.NoSpeech, 5},
		{"Translate", sp.Translate, 6},
		{"Transcribe", sp.Transcribe, 7},
		{"Prev", sp.Prev, 8},
		{"SOLM", sp.SOLM, 9},
		{"NotSOT", sp.NotSOT, 10},
		{"TimestampBegin", sp.TimestampBegin, 13},
		{"TimestampEnd", sp.TimestampEnd, 15},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("Special().%s = %d, want %d", c.name, c.got, c.want)
		}
	}

	if len(sp.Languages) != 2 {
		t.Errorf("len(Languages) = %d, want 2", len(sp.Languages))
	}
}

func TestNew_LengthMismatch(t *testing.T) {
	_, err := New([]string{"a", "b"}, []uint32{0})
	if err == nil {
		t.Fatal("expected error for mismatched lengths, got nil")
	}
}

func TestEncodeDecode_ASCII(t *testing.T) {
	v := buildTestVocab(t)

	tests := []struct {
		text   string
		tokens []Token
	}{
		{"hi", []Token{0}},
		{"Hello", []Token{1}},
	}

	for _, tt := range tests {
		enc := v.Encode(tt.text)
		if len(enc) != len(tt.tokens) {
			t.Errorf("Encode(%q) = %v, want %v", tt.text, enc, tt.tokens)
			continue
		}
		for i, tok := range enc {
			if tok != tt.tokens[i] {
				t.Errorf("Encode(%q)[%d] = %d, want %d", tt.text, i, tok, tt.tokens[i])
			}
		}

		got := v.Decode(enc)
		if got != tt.text {
			t.Errorf("Decode(Encode(%q)) = %q, want %q", tt.text, got, tt.text)
		}
	}
}

func TestDecode_SkipsSpecialTokens(t *testing.T) {
	v := buildTestVocab(t)
	// Mix a normal token with special tokens; special ones should be omitted.
	tokens := []Token{3 /*SOT*/, 0 /*hi*/, 4 /*EOT*/}
	got := v.Decode(tokens)
	if got != "hi" {
		t.Errorf("Decode with specials = %q, want %q", got, "hi")
	}
}

func TestDecodeToken(t *testing.T) {
	v := buildTestVocab(t)
	// Special tokens should decode to their string representation.
	got := v.DecodeToken(4 /*EOT*/)
	if got != "<|endoftext|>" {
		t.Errorf("DecodeToken(EOT) = %q, want %q", got, "<|endoftext|>")
	}
	// Normal token round-trip.
	got = v.DecodeToken(0 /*hi*/)
	if got != "hi" {
		t.Errorf("DecodeToken(0) = %q, want %q", got, "hi")
	}
}

func TestIsTimestamp(t *testing.T) {
	v := buildTestVocab(t)

	tests := []struct {
		token Token
		want  bool
	}{
		{13, true},  // <|0.00|>
		{14, true},  // <|1.50|>
		{15, true},  // <|30.00|>
		{0, false},  // hi
		{3, false},  // SOT
		{12, false}, // <|fr|>
	}
	for _, tt := range tests {
		if got := v.IsTimestamp(tt.token); got != tt.want {
			t.Errorf("IsTimestamp(%d) = %v, want %v", tt.token, got, tt.want)
		}
	}
}

func TestTimestampToMs(t *testing.T) {
	v := buildTestVocab(t)

	tests := []struct {
		token Token
		want  int64
	}{
		{13, 0},     // <|0.00|>   → 0 ms
		{14, 1500},  // <|1.50|>   → 1500 ms
		{15, 30000}, // <|30.00|>  → 30000 ms
		{0, 0},      // non-timestamp → 0
	}
	for _, tt := range tests {
		if got := v.TimestampToMs(tt.token); got != tt.want {
			t.Errorf("TimestampToMs(%d) = %d, want %d", tt.token, got, tt.want)
		}
	}
}

func TestLanguageID(t *testing.T) {
	v := buildTestVocab(t)

	tests := []struct {
		lang  string
		want  Token
		found bool
	}{
		{"en", 11, true},
		{"fr", 12, true},
		{"de", 0, false}, // not in test vocab
	}
	for _, tt := range tests {
		got, ok := v.LanguageID(tt.lang)
		if ok != tt.found {
			t.Errorf("LanguageID(%q) found=%v, want %v", tt.lang, ok, tt.found)
			continue
		}
		if ok && got != tt.want {
			t.Errorf("LanguageID(%q) = %d, want %d", tt.lang, got, tt.want)
		}
	}
}

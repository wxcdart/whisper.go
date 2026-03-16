package output

import "fmt"

// msToSRT formats milliseconds as HH:MM:SS,mmm (SRT format).
func msToSRT(ms int64) string {
	h := ms / 3_600_000
	ms %= 3_600_000
	m := ms / 60_000
	ms %= 60_000
	s := ms / 1_000
	ms %= 1_000
	return fmt.Sprintf("%02d:%02d:%02d,%03d", h, m, s, ms)
}

// msToVTT formats milliseconds as HH:MM:SS.mmm (WebVTT / plain-text format).
func msToVTT(ms int64) string {
	h := ms / 3_600_000
	ms %= 3_600_000
	m := ms / 60_000
	ms %= 60_000
	s := ms / 1_000
	ms %= 1_000
	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}

// msToLRC formats milliseconds as MM:SS.xx (centisecond precision, LRC format).
func msToLRC(ms int64) string {
	m := ms / 60_000
	ms %= 60_000
	s := ms / 1_000
	cs := (ms % 1_000) / 10
	return fmt.Sprintf("%02d:%02d.%02d", m, s, cs)
}

// msToSec converts milliseconds to seconds as float64.
func msToSec(ms int64) float64 {
	return float64(ms) / 1000.0
}

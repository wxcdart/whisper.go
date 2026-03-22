package ml

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
)

var (
	quantDebugOnce    sync.Once
	quantDebugEnabled bool
)

func quantDebugf(format string, args ...any) {
	quantDebugOnce.Do(func() {
		v := strings.TrimSpace(strings.ToLower(os.Getenv("WHISPERGO_QUANT_DEBUG")))
		quantDebugEnabled = v == "1" || v == "true" || v == "yes" || v == "on"
	})
	if !quantDebugEnabled {
		return
	}
	log.Printf("[ml:quant] %s", fmt.Sprintf(format, args...))
}

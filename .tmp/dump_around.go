package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	b, err := os.ReadFile("ggml-tiny.bin")
	if err != nil {
		panic(err)
	}
	needle := "encoder.conv1.bias"
	idx := strings.Index(string(b), needle)
	if idx < 0 {
		fmt.Println("not found")
		return
	}
	fmt.Printf("found at %d\n", idx)
	start := idx - 32
	if start < 0 {
		start = 0
	}
	end := idx + len(needle) + 128
	if end > len(b) {
		end = len(b)
	}
	chunk := b[start:end]
	for i := 0; i < len(chunk); i += 16 {
		line := chunk[i:]
		if len(line) > 16 {
			line = line[:16]
		}
		fmt.Printf("%08x  ", start+i)
		for j := 0; j < len(line); j++ {
			fmt.Printf("%02x ", line[j])
		}
		for j := len(line); j < 16; j++ {
			fmt.Print("   ")
		}
		fmt.Print(" ")
		for j := 0; j < len(line); j++ {
			c := line[j]
			if c < 32 || c > 126 {
				fmt.Print('.')
			} else {
				fmt.Printf("%c", c)
			}
		}
		fmt.Println()
	}
}

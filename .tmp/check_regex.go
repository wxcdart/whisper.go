package main

import (
	"fmt"
	"io"
	"os"
	"regexp"
)

func main(){
	f, err := os.Open("ggml-tiny.bin")
	if err!=nil{panic(err)}
	defer f.Close()
	fi, _ := f.Stat()
	buf := make([]byte, fi.Size())
	if _, err := io.ReadFull(f, buf); err!=nil{panic(err)}
	re := regexp.MustCompile("[A-Za-z0-9_.]{6,200}")
	matches := re.FindAllString(string(buf), -1)
	fmt.Printf("matches: %d\n", len(matches))
	// print first 40 unique
	seen := make(map[string]bool)
	count:=0
	for _, m := range matches{
		if !seen[m]{
			seen[m]=true
			fmt.Println(m)
			count++
			if count>=60{break}
		}
	}
}

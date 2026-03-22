package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

func main(){
	f, err := os.Open("ggml-tiny.bin")
	if err!=nil{panic(err)}
	defer f.Close()
	fi, _ := f.Stat()
	buf := make([]byte, fi.Size())
	if _, err := io.ReadFull(f, buf); err!=nil{panic(err)}
	re := regexp.MustCompile("[A-Za-z0-9_.]{6,200}")
	matches := re.FindAllIndex(buf, -1)
	count:=0
	for _, m := range matches {
		name := string(buf[m[0]:m[1]])
		if !strings.Contains(name, "encoder") { continue }
		// try parse descriptor after end
		for offShift:=0; offShift<8; offShift++{
			p := m[1]+offShift
			if p+4>len(buf){continue}
			ndim := binary.LittleEndian.Uint32(buf[p:p+4])
			if ndim==0 || ndim>8 { continue }
			q:=p+4
			okShape:=true
			shape := make([]uint32, ndim)
			for i:=uint32(0); i<ndim; i++{
				if q+4>len(buf){okShape=false; break}
				d := binary.LittleEndian.Uint32(buf[q:q+4])
				if d==0 || d>1000000 { okShape=false; break }
				shape[i]=d
				q+=4
			}
			if !okShape{ continue }
			if q+4>len(buf){continue}
			dtype := binary.LittleEndian.Uint32(buf[q:q+4]); q+=4
			if q+8>len(buf){continue}
			off := binary.LittleEndian.Uint64(buf[q:q+8])
			fmt.Printf("name=%q offShift=%d ndim=%d shape=%v dtype=%d off=%d\n", name, offShift, ndim, shape, dtype, off)
			count++
			if count>30{ return }
		}
	}
}

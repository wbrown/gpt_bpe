package main

import (
	"github.com/siongui/goef"
)

func main() {
	err := goef.GenerateGoPackage("resources", "../embedded",
		"../resource_data_js.go")
	if err != nil {
		panic(err)
	}
}

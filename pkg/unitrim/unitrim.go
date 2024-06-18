package unitrim

import (
	"encoding/json"
	"github.com/wbrown/gpt_bpe"
	"log"
	"os"
	"path"
)

func AppendUnitrimJSON(dir string) {
	// read in the llama2 encoder file
	encoderBytes, err := os.ReadFile(path.Join(dir, "vocab.json"))
	if err != nil {
		log.Fatalf("Could not read encoder file: %v\n", err)
	}
	// unmarshal the encoder file
	var encoder map[string]gpt_bpe.Token
	err = json.Unmarshal(encoderBytes, &encoder)
	if err != nil {
		log.Fatalf("Could not unmarshal encoder file: %v\n", err)
	}

	// get generated array for unitrim with the makeUnitrimArr function
	generatedArray := gpt_bpe.MakeUnitrimArr(encoder)

	// write the generated array to a file
	unitrimFile := path.Join(dir, "unitrim.json")
	unitrimBytes, err := json.Marshal(generatedArray)
	if err != nil {
		log.Fatalf("Could not marshal generated array: %v\n", err)
	}
	err = os.WriteFile(unitrimFile, unitrimBytes, 0644)
	if err != nil {
		log.Fatalf("Could not write unitrim file: %v\n", err)
	}
}

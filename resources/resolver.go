package resources

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"io/ioutil"
	"log"
	"net/url"
	"os"
	"path"
	"time"

	"github.com/dustin/go-humanize"
)

type ResourceFlag uint8
type ResourceType uint8

// WriteCounter counts the number of bytes written to it, and every 10 seconds,
// it prints a message reporting the number of bytes written so far.
type WriteCounter struct {
	Total    uint64
	Last     time.Time
	Reported bool
	Path     string
	Size     uint64
}

func (wc *WriteCounter) Write(p []byte) (int, error) {
	n := len(p)
	wc.Total += uint64(n)
	if time.Now().Sub(wc.Last).Seconds() > 10 {
		wc.Reported = true
		wc.Last = time.Now()
		log.Print(fmt.Sprintf("Downloading %s... %s / %s completed.",
			wc.Path, humanize.Bytes(wc.Total), humanize.Bytes(wc.Size)))
	}
	return n, nil
}

// Enumeration of resource flags that indicate what the resolver should do
// with the resource.
const (
	RESOURCE_REQUIRED ResourceFlag = 1 << iota
	RESOURCE_OPTIONAL
	RESOURCE_DERIVED
	RESOURCE_MODEL
	RESOURCE_ONEOF
)

// Enumeration of different types of models
const (
	RESOURCETYPE_TRANSFORMERS ResourceType = 1 << iota
	RESOURCETYPE_DIFFUSERS
)

type ResourceEntryDefs map[string]ResourceFlag
type ResourceEntry struct {
	file interface{}
	Data *[]byte
}

type Resources map[string]ResourceEntry

func (rsrcs *Resources) Cleanup() {
	for _, rsrc := range *rsrcs {
		file := rsrc.file
		switch t := file.(type) {
		case os.File:
			t.Close()
		case fs.File:
			t.Close()
		}
	}
}

// GetResourceEntries
// Returns a default map of resource entries that express what files are
// required, optional, derived, and/or model resources. Requires a resourcetype.
func GetResourceEntries(typ ResourceType) ResourceEntryDefs {
	switch typ {
	case RESOURCETYPE_TRANSFORMERS:
		return ResourceEntryDefs{
			"config.json":             RESOURCE_REQUIRED,
			"vocab.json":              RESOURCE_REQUIRED,
			"merges.txt":              RESOURCE_REQUIRED,
			"special_tokens_map.json": RESOURCE_OPTIONAL,
			"unitrim.json":            RESOURCE_OPTIONAL,
			"wordtokens.json":         RESOURCE_OPTIONAL,
			"specials.txt":            RESOURCE_OPTIONAL | RESOURCE_DERIVED,
			"tokenizer_config.json":   RESOURCE_OPTIONAL,
			"tokenizer.json":          RESOURCE_OPTIONAL,
			"pytorch_model.bin":       RESOURCE_MODEL,
		}
	case RESOURCETYPE_DIFFUSERS:
		return ResourceEntryDefs{
			"feature_extractor/preprocessor_config.json": RESOURCE_REQUIRED,
			"safety_checker/config.json":                 RESOURCE_OPTIONAL,
			"safety_checker/pytorch_model.json":          RESOURCE_OPTIONAL,
			"scheduler/scheduler_config.json":            RESOURCE_OPTIONAL,
			"text_encoder/config.json":                   RESOURCE_REQUIRED,
			"text_encoder/pytorch_model.bin":             RESOURCE_MODEL,
			"tokenizer/merges.txt":                       RESOURCE_REQUIRED,
			"tokenizer/special_tokens_map.json":          RESOURCE_REQUIRED,
			"tokenizer/tokenizer_config.json":            RESOURCE_REQUIRED,
			"tokenizer/vocab.json":                       RESOURCE_REQUIRED,
			"unet/config.json":                           RESOURCE_REQUIRED,
			"unet/diffusion_pytorch_model.bin":           RESOURCE_MODEL,
			"vae/config.json":                            RESOURCE_REQUIRED,
			"vae/diffusion_pytorch_model.bin":            RESOURCE_MODEL,
		}
	default:
		return ResourceEntryDefs{}
	}
}

// FetchHuggingFace
// Wrapper around FetchHTTP that fetches a resource from huggingface.co.
func FetchHuggingFace(id string, rsrc string) (io.ReadCloser, error) {
	token := os.Getenv("HF_API_TOKEN")
	return FetchHTTP("https://huggingface.co/"+id+"/resolve/main", rsrc, token)
}

// SizeHuggingFace
// Wrapper around SizeHTTP that gets the size of a resource from huggingface.co.
func SizeHuggingFace(id string, rsrc string) (uint, error) {
	token := os.Getenv("HF_API_TOKEN")
	return SizeHTTP("https://huggingface.co/"+id+"/resolve/main", rsrc, token)
}

func isValidUrl(toTest string) bool {
	_, err := url.ParseRequestURI(toTest)
	if err != nil {
		return false
	}

	u, err := url.Parse(toTest)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return false
	}

	return true
}

// Fetch
// Given a base URI and a resource name, determines if the resource is local,
// remote, or from huggingface.co. If the resource is local, it returns a
// file handle to the resource. If the resource is remote, or from
// huggingface.co, it fetches the resource and returns a ReadCloser to the
// fetched or cached resource.
func Fetch(uri string, rsrc string, token string) (io.ReadCloser, error) {
	if isValidUrl(uri) {
		return FetchHTTP(uri, rsrc, token)
	} else if _, err := os.Stat(path.Join(uri, rsrc)); !os.IsNotExist(err) {
		if handle, fileErr := os.Open(path.Join(uri, rsrc)); fileErr != nil {
			return nil, errors.New(
				fmt.Sprintf("error opening %s/%s: %v",
					uri, rsrc, fileErr))
		} else {
			return handle, fileErr
		}
	} else {
		return FetchHuggingFace(uri, rsrc)
	}
}

// Size
// Given a base URI and a resource name, determine the size of the resource.
func Size(uri string, rsrc string, token string) (uint, error) {
	if isValidUrl(uri) {
		return SizeHTTP(uri, rsrc, token)
	} else if fsz, err := os.Stat(path.Join(uri, rsrc)); !os.IsNotExist(err) {
		return uint(fsz.Size()), nil
	} else {
		return SizeHuggingFace(uri, rsrc)
	}
}

// AddEntry
// Add a resource to the Resources map, opening it as a mmap.Map.
func (rsrcs *Resources) AddEntry(name string, file *os.File) error {
	fileMmap, mmapErr := readMmap(file)
	if mmapErr != nil {
		return errors.New(
			fmt.Sprintf("error trying to mmap file: %s",
				mmapErr))
	} else {
		(*rsrcs)[name] = ResourceEntry{file, fileMmap}
	}
	return nil
}

// Specials
// Map of special tokens such as `<|pad|>`, `<|endoftext|>`, etc.
type Specials map[string]string

// ResolveSpecialTokens
// If `specials.json` does not exist in `dir`, create it from the
// `special_tokens_map.json` file.
func (rsrcs *Resources) ResolveSpecialTokens(dir string) (
	realizedSpecials Specials, err error) {
	realizedSpecials = make(Specials, 0)
	// If we already have `specials.json`, we don't need to generate it.
	if _, ok := (*rsrcs)["specials.json"]; ok {
		if specErr := json.Unmarshal(*(*rsrcs)["specials.json"].Data,
			&realizedSpecials); specErr != nil {
			return nil, errors.New(
				fmt.Sprintf("cannot unmarshal `specials.json`: %s",
					specErr))
		}
		return realizedSpecials, nil
	}

	// We can only generate `specials.json` if we have `special_tokens_map`
	specialsJson, ok := (*rsrcs)["special_tokens_map.json"]
	if !ok {
		return nil, nil
	}

	specialTokens := make(map[string]interface{}, 0)
	if specialErr := json.Unmarshal(*specialsJson.Data,
		&specialTokens); specialErr != nil {
		return nil, specialErr
	}

	for k, v := range specialTokens {
		var specialToken string
		switch t := v.(type) {
		case string:
			specialToken = t
		case map[string]interface{}:
			mv := t["content"]
			switch mvt := mv.(type) {
			case string:
				specialToken = mvt
			default:
				log.Fatal(fmt.Sprintf("unknown format for `special_tokens_map."+
					"json`: %v", t))
			}
		default:
			log.Fatal(fmt.Sprintf("unknown format for `special_tokens_map."+
				"json`: %v", t))
		}
		realizedSpecials[k] = specialToken
	}
	if len(realizedSpecials) > 0 {
		specialsFile, specialFileErr := os.OpenFile(
			path.Join(dir, "specials.json"),
			os.O_TRUNC|os.O_RDWR|os.O_CREATE, 0755)
		if specialFileErr != nil {
			return nil, errors.New(
				fmt.Sprintf("cannot generate `specials.json`: %s",
					specialFileErr))
		}
		specialsJsonBytes, specialsErr := json.Marshal(realizedSpecials)
		if specialsErr != nil {
			specialsFile.Close()
			return nil, errors.New(
				fmt.Sprintf("cannot marshal `specials.json`: %s",
					specialsErr))
		}
		if _, writeErr := specialsFile.Write(
			specialsJsonBytes); writeErr != nil {
			specialsFile.Close()
			return nil, errors.New(
				fmt.Sprintf("cannot write `specials.json`: %s",
					specialsErr))
		}
		if _, seekErr := specialsFile.Seek(0, 0); seekErr != nil {
			return nil, errors.New(
				fmt.Sprintf("cannot seek `specials.json`: %s",
					seekErr))
		}
		if mmapErr := rsrcs.AddEntry("specials.json",
			specialsFile); mmapErr != nil {
			return nil, mmapErr
		}
	}
	return realizedSpecials, nil
}

// ResolveResources resolves all resources at a given uri, and checks if they
// exist in the given directory. If they don't exist, they are downloaded.
func ResolveResources(
	uri string,
	dir *string,
	rsrcLvl ResourceFlag,
	rsrcType ResourceType,
	token string,
) (
	*Resources,
	error,
) {
	foundResources := make(Resources, 0)
	resources := GetResourceEntries(rsrcType)

	for file, flag := range resources {
		var rsrcFile os.File
		if flag <= rsrcLvl {
			log.Printf("Resolving %s/%s... ", uri, file)
			targetPath := path.Join(*dir, file)
			rsrcSize, rsrcSizeErr := Size(uri, file, token)
			if rsrcSizeErr != nil {
				if flag&RESOURCE_REQUIRED != 0 {
					log.Printf("%s/%s not found, required!",
						uri, file)
					return &foundResources, errors.New(
						fmt.Sprintf(
							"cannot retrieve required `%s` from `%s`: %s",
							uri, file, rsrcSizeErr))
				} else {
					log.Printf("Resolved %s/%s... not there, not required.",
						uri, file)
					continue
				}
			} else if targetStat, targetStatErr := os.Stat(targetPath); !os.IsNotExist(
				targetStatErr) && uint(targetStat.Size()) == rsrcSize {
				log.Printf("Skipping %s/%s... already exists, "+
					"and of the correct size.", uri, file)
				openFile, skipFileErr := os.OpenFile(
					path.Join(*dir, file),
					os.O_RDONLY, 0755)
				if skipFileErr != nil {
					return &foundResources, errors.New(
						fmt.Sprintf("error opening '%s' for write: %s",
							file, skipFileErr))
				} else {
					rsrcFile = *openFile
				}
			} else if rsrcReader, rsrcErr := Fetch(uri, file, token); rsrcErr != nil {
				return &foundResources, errors.New(
					fmt.Sprintf(
						"cannot retrieve `%s` from `%s`: %s",
						uri, file, rsrcErr))
			} else {
				if dirErr := os.MkdirAll(
					path.Dir(path.Join(*dir, file)), 0755); dirErr != nil {
					return &foundResources, errors.New(
						fmt.Sprintf("cannot create directory for '%s': %s",
							file, dirErr))
				}
				openFile, rsrcFileErr := os.OpenFile(
					path.Join(*dir, file),
					os.O_TRUNC|os.O_RDWR|os.O_CREATE, 0755)
				rsrcFile = *openFile
				if rsrcFileErr != nil {
					return &foundResources, errors.New(
						fmt.Sprintf("error opening '%s' for write: %s",
							file, rsrcFileErr))
				}
				counter := &WriteCounter{
					Last: time.Now(),
					Path: fmt.Sprintf("%s/%s", uri, file),
					Size: uint64(rsrcSize),
				}
				bytesDownloaded, ioErr := io.Copy(&rsrcFile,
					io.TeeReader(rsrcReader, counter))
				rsrcReader.Close()
				if ioErr != nil {
					return &foundResources, errors.New(
						fmt.Sprintf("error downloading '%s': %s",
							file, ioErr))
				} else {
					log.Println(fmt.Sprintf("Downloaded %s/%s... "+
						"%s completed.", uri, file,
						humanize.Bytes(uint64(bytesDownloaded))))
				}
			}
			if mmapErr := foundResources.AddEntry(file,
				&rsrcFile); mmapErr != nil {
				return &foundResources, errors.New(
					fmt.Sprintf("error trying to mmap file: %s",
						mmapErr))
			}
		}
	}
	return &foundResources, nil
}

// HFConfig contains the tokenizer configuration that gpt_bpe uses.
type HFConfig struct {
	ModelId        *string `json:"omitempty"`
	ModelType      *string `json:"model_type,omitempty"`
	EosTokenId     *uint16 `json:"eos_token_id,omitempty"`
	BosTokenId     *uint16 `json:"bos_token_id,omitempty"`
	PadTokenId     *uint16 `json:"pad_token_id,omitempty"`
	BosTokenStr    *string `json:"bos_token,omitempty"`
	EosTokenStr    *string `json:"eos_token,omitempty"`
	PadTokenStr    *string `json:"pad_token,omitempty"`
	VocabSize      *uint16 `json:"vocab_size,omitempty"`
	Newlinemode    *string `json:"newlinemode,omitempty"`
	TokenizerClass *string `json:"tokenizer_class"`
}

// Additional special tokenizer configuration.
type SpecialConfig struct {
	PuncRunes     []*string          `json:"punc_runes"`
	Normalizer    *map[string]string `json:"normalizer"`
	EncloseEosBos bool               `json:"enclose_eos_bos"`
	PrefixSpace   bool               `json:"prefix_space"`
	LowerCase     bool               `json:"lower_case"`
	EndOfWord     string             `json:"end_of_word"`
}

// ResolveConfig
// Resolves a given vocabulary id, and returns the corresonding HuggingFace
// configuration, and the resources for the tokenizer.
func ResolveConfig(vocabId string, token string) (config *HFConfig,
	resources *Resources, err error) {
	dir, dirErr := ioutil.TempDir("", "resources")
	if dirErr != nil {
		return nil, nil, dirErr
	}
	defer os.RemoveAll(dir)
	rslvdResources, rsrcErr := ResolveResources(
		vocabId,
		&dir,
		RESOURCE_DERIVED,
		RESOURCETYPE_TRANSFORMERS,
		token)
	if rsrcErr != nil {
		return nil, nil, rsrcErr
	} else {
		resources = rslvdResources
	}

	var hfConfig HFConfig
	if configErr := json.Unmarshal(*((*resources)["config.json"]).Data,
		&hfConfig); configErr != nil {
		resources.Cleanup()
		return nil, nil, errors.New(fmt.Sprintf(
			"error unmarshalling `config.json`: %s", configErr))
	}

	specialTokens, specialsErr := resources.ResolveSpecialTokens(dir)
	if specialsErr != nil {
		resources.Cleanup()
		return nil, nil, specialsErr
	}
	defaultTkn := "<|endoftext|>"
	eosToken, ok := specialTokens["eos_token"]
	if !ok {
		eosToken = defaultTkn
	}
	hfConfig.EosTokenStr = &eosToken
	padToken, ok := specialTokens["pad_token"]
	if !ok {
		padToken = defaultTkn
	}
	hfConfig.PadTokenStr = &padToken
	bosToken, ok := specialTokens["bos_token"]
	if !ok {
		bosToken = defaultTkn
	}
	hfConfig.BosTokenStr = &bosToken

	if hfConfig.EosTokenStr == nil {
		hfConfig.EosTokenStr = &defaultTkn
	}
	if hfConfig.PadTokenStr == nil {
		hfConfig.PadTokenStr = &defaultTkn
	}
	if hfConfig.BosTokenStr == nil {
		hfConfig.BosTokenStr = &defaultTkn
	}

	return &hfConfig, resources, nil
}

// ResolveVocabId
// Resolves a vocabulary id to a set of resources, from embedded,
// local filesystem, or remote.
func ResolveVocabId(vocabId string, token string) (*HFConfig, *Resources, error) {
	var resolvedVocabId string
	if _, vocabErr := EmbeddedDirExists(vocabId); vocabErr == nil {
		endOfText := "<|endoftext|>"
		bosText := "<|startoftext|>"
		hf := &HFConfig{
			ModelId:     &vocabId,
			BosTokenStr: &bosText,
			EosTokenStr: &endOfText,
			PadTokenStr: &endOfText,
		}
		resources := make(Resources, 0)
		if unitrim := GetEmbeddedResource(vocabId + "/unitrim." +
			"json"); unitrim != nil {
			resources["unitrim.json"] = *unitrim
		}
		if config := GetEmbeddedResource(vocabId + "/encoder." +
			"json"); config != nil {
			resources["vocab.json"] = *config
		}
		if vocab := GetEmbeddedResource(vocabId + "/vocab.bpe"); vocab != nil {
			resources["merges.txt"] = *vocab
		}
		if specials_t := GetEmbeddedResource(vocabId + "/specials.txt"); specials_t != nil {
			resources["specials.txt"] = *specials_t
		}
		if specials := GetEmbeddedResource(vocabId + "/special_tokens_map." +
			"json"); specials != nil {
			resources["special_tokens_map.json"] = *specials
		}
		special_config := GetEmbeddedResource(vocabId + "/special_config.json")
		if special_config != nil {
			resources["special_config.json"] = *special_config
		}
		return hf, &resources, nil
	} else {
		log.Printf("%v", vocabErr)
	}
	if isValidUrl(vocabId) {
		u, _ := url.Parse(vocabId)
		basePath := path.Base(u.Path)
		resolvedVocabId = basePath
	} else {
		resolvedVocabId = vocabId
	}
	config, resources, err := ResolveConfig(vocabId, token)
	if err != nil {
		return nil, nil, err
	} else {
		config.ModelId = &resolvedVocabId
		if _, exists := (*resources)["unitrim.json"]; !exists {
			(*resources)["unitrim.json"] = *GetEmbeddedResource(
				"gpt2-tokenizer/unitrim.json")
		}
		return config, resources, nil
	}
}

func DestructureTokenizerJSON(dir string) {
	//This function is used to destruct the tokenizer.json
	//file into the vocab.json and merges.txt files
	//This is done if HF uses the new tokenizer.json format

	//We get the Directory of all the files
	tokenizerPath := path.Join(dir, "tokenizer.json")
	mergesPath := path.Join(dir, "merges.txt")
	vocabPath := path.Join(dir, "vocab.json")

	// Open the file
	file, err := os.Open(tokenizerPath)
	if err != nil {
		log.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	// Decode the JSON data into a map
	var data map[string]interface{}
	err = json.NewDecoder(file).Decode(&data)
	if err != nil {
		log.Println("Error decoding JSON:", err)
		return
	}

	// Access the data at the specified path
	model, ok := data["model"].(map[string]interface{})
	if !ok {
		log.Println("Error: Could not convert value to map")
		return
	}
	vocab, ok := model["vocab"].(map[string]interface{})
	if !ok {
		log.Println("Error: Could not convert value to map")
		return
	}

	merges, ok := model["merges"].([]interface{})
	if !ok {
		log.Println("Error: Could not convert value to map")
		return
	}

	// Convert the slice of interfaces to a slice of strings
	var mergesStr []string
	for _, v := range merges {
		mergesStr = append(mergesStr, v.(string))
	}

	// Marshal the vocab map into a JSON string with indentation
	b_vocab, err := json.MarshalIndent(vocab, "", "  ")
	if err != nil {
		log.Println("Error marshaling JSON:", err)
		return
	}

	// Create the file
	vocabfile, err := os.Create(vocabPath)
	if err != nil {
		log.Println("Error creating file:", err)
		return
	}
	defer vocabfile.Close()

	// Write the JSON string to the file
	_, err = vocabfile.Write(b_vocab)
	if err != nil {
		log.Println("Error writing to file:", err)
		return
	}

	log.Println("Vocab written to vocab.json from tokenizer.json")

	// Create the file
	mergesfile, err := os.Create(mergesPath)
	if err != nil {
		log.Println("Error creating file:", err)
		return
	}
	defer mergesfile.Close()

	// Write each merge string to a new line in the file
	for _, v := range merges {
		_, err = mergesfile.WriteString(v.(string) + "\n")
		if err != nil {
			log.Println("Error writing to file:", err)
			return
		}
	}

	log.Println("Merges written to merges.txt from tokenizer.json")
}

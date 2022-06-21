package resources

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/dustin/go-humanize"
	mmap "github.com/edsrzf/mmap-go"
	"io"
	"io/fs"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"strconv"
	"time"
)

type ResourceFlag uint8

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

func (wc WriteCounter) PrintProgress() {
	// Clear the line by using a character return to go back to the start and remove
	// the remaining characters by filling it with spaces
	// Return again and print current status of download
	// We use the humanize package to print the bytes in a meaningful way (e.g. 10 MB)
}

const (
	RESOURCE_REQUIRED ResourceFlag = 1 << iota
	RESOURCE_OPTIONAL
	RESOURCE_DERIVED
	RESOURCE_MODEL
	RESOURCE_ONEOF
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

func GetResourceEntries() ResourceEntryDefs {
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
}

func FetchHTTP(uri string, rsrc string) (io.ReadCloser, error) {
	resp, remoteErr := http.Get(uri + "/" + rsrc)
	if remoteErr != nil {
		return nil, remoteErr
	} else if resp.StatusCode != 200 {
		return nil, errors.New(fmt.Sprintf("HTTP status code %d",
			resp.StatusCode))
	}
	return resp.Body, nil
}

func SizeHTTP(uri string, rsrc string) (uint, error) {
	resp, remoteErr := http.Head(uri + "/" + rsrc)
	if remoteErr != nil {
		return 0, remoteErr
	} else if resp.StatusCode != 200 {
		return 0, errors.New(fmt.Sprintf("HTTP status code %d",
			resp.StatusCode))
	} else {
		size, _ := strconv.Atoi(resp.Header.Get("Content-Length"))
		return uint(size), nil
	}
}

func FetchHuggingFace(id string, rsrc string) (io.ReadCloser, error) {
	return FetchHTTP("https://huggingface.co/"+id+"/resolve/main", rsrc)
}

func SizeHuggingFace(id string, rsrc string) (uint, error) {
	return SizeHTTP("https://huggingface.co/"+id+"/resolve/main", rsrc)
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

func Fetch(uri string, rsrc string) (io.ReadCloser, error) {
	if isValidUrl(uri) {
		return FetchHTTP(uri, rsrc)
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

func Size(uri string, rsrc string) (uint, error) {
	if isValidUrl(uri) {
		return SizeHTTP(uri, rsrc)
	} else if fsz, err := os.Stat(path.Join(uri, rsrc)); !os.IsNotExist(err) {
		return uint(fsz.Size()), nil
	} else {
		return SizeHuggingFace(uri, rsrc)
	}
}

func (rsrcs *Resources) AddEntry(name string, file *os.File) error {
	fileMmap, mmapErr := mmap.Map(file, mmap.RDONLY, 0)
	if mmapErr != nil {
		return errors.New(
			fmt.Sprintf("error trying to mmap file: %s",
				mmapErr))
	} else {
		(*rsrcs)[name] = ResourceEntry{file, (*[]byte)(&fileMmap)}
	}
	return nil
}

type Specials map[string]string

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

func ResolveResources(uri string, dir *string,
	rsrcLvl ResourceFlag) (*Resources,
	error) {
	foundResources := make(Resources, 0)
	resources := GetResourceEntries()

	for file, flag := range resources {
		var rsrcFile os.File
		if flag <= rsrcLvl {
			log.Printf("Resolving %s/%s... ", uri, file)
			targetPath := path.Join(*dir, file)
			rsrcSize, rsrcSizeErr := Size(uri, file)
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
			} else if rsrcReader, rsrcErr := Fetch(uri, file); rsrcErr != nil {
				return &foundResources, errors.New(
					fmt.Sprintf(
						"cannot retrieve `%s` from `%s`: %s",
						uri, file, rsrcErr))
			} else {
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

type HFConfig struct {
	ModelId        *string `json:"omitempty"`
	ModelType      *string `json:"model_type,omitempty"`
	EosTokenId     *uint16 `json:"eos_token_id,omitempty"`
	BosTokenId     *uint16 `json:"bos_token_id,omitempty"`
	PadTokenId     *uint16 `json:"pad_token_id,omitempty"`
	EosTokenStr    *string `json:"eos_token,omitempty"`
	PadTokenStr    *string `json:"pad_token,omitempty"`
	VocabSize      *uint16 `json:"vocab_size,omitempty"`
	Newlinemode    *string `json:"newlinemode,omitempty"`
	TokenizerClass *string `json:"tokenizer_class"`
}

func ResolveConfig(vocabId string) (config *HFConfig,
	resources *Resources, err error) {
	dir, dirErr := ioutil.TempDir("", "resources")
	if dirErr != nil {
		return nil, nil, dirErr
	}
	defer os.RemoveAll(dir)
	rslvdResources, rsrcErr := ResolveResources(vocabId, &dir, RESOURCE_DERIVED)
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

	if hfConfig.EosTokenStr == nil {
		hfConfig.EosTokenStr = &defaultTkn
	}
	if hfConfig.PadTokenStr == nil {
		hfConfig.PadTokenStr = &defaultTkn
	}

	return &hfConfig, resources, nil
}

func GetCompiledResource(path string) ResourceEntry {
	resourceFile, _ := f.Open(path)
	resourceBytes, _ := f.ReadFile(path)
	return ResourceEntry{&resourceFile, &resourceBytes}
}

func ResolveVocabId(vocabId string) (*HFConfig, *Resources, error) {
	var resolvedVocabId string
	if _, vocabErr := f.ReadDir(vocabId); vocabErr == nil {
		endOfText := "<|endoftext|>"
		hf := &HFConfig{
			ModelId:     &vocabId,
			EosTokenStr: &endOfText,
			PadTokenStr: &endOfText,
		}
		resources := make(Resources, 0)
		resources["unitrim.json"] = GetCompiledResource(
			vocabId + "/unitrim.json")
		resources["vocab.json"] = GetCompiledResource(
			vocabId + "/encoder.json")
		resources["merges.txt"] = GetCompiledResource(
			vocabId + "/vocab.bpe")
		resources["specials.txt"] = GetCompiledResource(
			vocabId + "/specials.txt")
		return hf, &resources, nil
	}
	if isValidUrl(vocabId) {
		u, _ := url.Parse(vocabId)
		basePath := path.Base(u.Path)
		resolvedVocabId = basePath
	} else {
		resolvedVocabId = vocabId
	}
	config, resources, err := ResolveConfig(vocabId)
	if err != nil {
		return nil, nil, err
	} else {
		config.ModelId = &resolvedVocabId
		if _, exists := (*resources)["unitrim.json"]; !exists {
			(*resources)["unitrim.json"] = GetCompiledResource(
				"gpt2-tokenizer/unitrim.json")
		}
		return config, resources, nil
	}
}

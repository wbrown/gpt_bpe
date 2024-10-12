package resources

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"io/ioutil"
	"log"
	"math"
	"net/url"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/wbrown/gpt_bpe/types"

	"github.com/dustin/go-humanize"
)

type Token types.Token
type Tokens types.Tokens
type TokenMap types.TokenMap

type JsonMap map[string]interface{}

type ResourceFlag uint8
type ResourceType uint8

// WriteCounter counts the number of bytes written to it, and every 10
// seconds, it prints a message reporting the number of bytes written so far.
type WriteCounter struct {
	Total    uint64
	Last     time.Time
	Reported bool
	Path     string
	Size     uint64
}

// Write writes p to the WriteCounter and updates the total number of bytes
// written.
func (wc *WriteCounter) Write(p []byte) (int, error) {
	n := len(p)
	wc.Total += uint64(n)
	if time.Since(wc.Last).Seconds() > 10 {
		wc.Reported = true
		wc.Last = time.Now()
		log.Printf(
			"Downloading %s... %s / %s completed.",
			wc.Path, humanize.Bytes(wc.Total), humanize.Bytes(wc.Size),
		)
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

var TransformerResources = ResourceEntryDefs{
	"config.json":                  RESOURCE_REQUIRED,
	"vocab.json":                   RESOURCE_OPTIONAL,
	"merges.txt":                   RESOURCE_OPTIONAL,
	"special_tokens_map.json":      RESOURCE_OPTIONAL,
	"wordtokens.json":              RESOURCE_OPTIONAL,
	"specials.txt":                 RESOURCE_OPTIONAL | RESOURCE_DERIVED,
	"tokenizer_config.json":        RESOURCE_OPTIONAL,
	"pytorch_model.bin.index.json": RESOURCE_OPTIONAL,
	"tokenizer.json":               RESOURCE_OPTIONAL,
	"tokenizer.model":              RESOURCE_OPTIONAL,
	"pytorch_model.bin":            RESOURCE_MODEL,
}

var DiffuserResources = ResourceEntryDefs{
	"feature_extractor/preprocessor_config.json": RESOURCE_OPTIONAL,
	"safety_checker/config.json":                 RESOURCE_OPTIONAL,
	"safety_checker/pytorch_model.bin":           RESOURCE_OPTIONAL,
	"scheduler/scheduler_config.json":            RESOURCE_REQUIRED,
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
	"model_index.json":                           RESOURCE_REQUIRED,
}

type ResourceEntryDefs map[string]ResourceFlag
type ResourceEntry struct {
	file interface{}
	Data *[]byte
}

type Resources map[string]ResourceEntry

// Cleanup closes all open file handles in the Resources map.
func (rsrcs *Resources) Cleanup() {
	for _, rsrc := range *rsrcs {
		file := rsrc.file
		switch t := file.(type) {
		case os.File:
			_ = t.Close()
		case fs.File:
			_ = t.Close()
		}
	}
}

// GetFile
// Returns the file handle for a given resource name.
func (rsrcs *Resources) GetFile(name string) (interface{}, error) {
	if rsrcEntry, ok := (*rsrcs)[name]; ok {
		return rsrcEntry.file, nil
	} else {
		return nil, fmt.Errorf("file %s not found", name)
	}
}

// GetResourceEntries
// Returns a default map of resource entries that express what files are
// required, optional, derived, and/or model resources. Requires a
// ResourceType.
func GetResourceEntries(typ ResourceType) ResourceEntryDefs {
	switch typ {
	case RESOURCETYPE_TRANSFORMERS:
		return TransformerResources
	case RESOURCETYPE_DIFFUSERS:

		return DiffuserResources
	default:
		return ResourceEntryDefs{}
	}
}

// getResourceEntryAliases
// Returns a map of defined resources to known alternative filenames
// for each resource of a given ResourceType.
func getResourceEntryAliases(typ ResourceType) map[string][]string {
	switch typ {
	case RESOURCETYPE_TRANSFORMERS:
		return map[string][]string{
			"vocab.json": {"encoder.json"},
		}
	default:
		return map[string][]string{}
	}
}

// FetchHuggingFace
// Wrapper around FetchHTTP that fetches a resource from huggingface.co.
func FetchHuggingFace(id string, rsrc string) (io.ReadCloser, error) {
	token := os.Getenv("HF_API_TOKEN")
	return FetchHTTP(
		"https://huggingface.co/"+id+"/resolve/main", rsrc, token,
	)
}

// SizeHuggingFace
// Wrapper around SizeHTTP that gets the size of a resource from huggingface.co.
func SizeHuggingFace(id string, rsrc string) (uint, error) {
	token := os.Getenv("HF_API_TOKEN")
	return SizeHTTP("https://huggingface.co/"+id+"/resolve/main", rsrc, token)
}

// isValidUrl
// Checks if a given string is a valid URL, returns true if it is, false
// otherwise.
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
			return nil, fmt.Errorf(
				"error opening %s/%s: %v",
				uri, rsrc, fileErr,
			)
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
		return fmt.Errorf("error trying to mmap file: %s", mmapErr)
	} else {
		(*rsrcs)[name] = ResourceEntry{file, fileMmap}
	}
	return nil
}

// Specials
// Map of special tokens such as <|pad|>, <|endoftext|>, etc.
type Specials map[string]string

// ResolveSpecialTokens
// If specials.json does not exist in dir, create it from the
// special_tokens_map.json file.
func (rsrcs *Resources) ResolveSpecialTokens(dir string) (
	realizedSpecials Specials, err error,
) {
	realizedSpecials = make(Specials)
	// If we already have specials.json, we don't need to generate it.
	if _, ok := (*rsrcs)["specials.json"]; ok {
		if specErr := json.Unmarshal(
			*(*rsrcs)["specials.json"].Data,
			&realizedSpecials,
		); specErr != nil {
			return nil, fmt.Errorf(
				"cannot unmarshal specials.json: %s", specErr,
			)
		}
		return realizedSpecials, nil
	}

	// We can only generate specials.json if we have special_tokens_map
	specialsJson, ok := (*rsrcs)["special_tokens_map.json"]
	if !ok {
		return nil, nil
	}

	specialTokens := make(JsonMap)
	if specialErr := json.Unmarshal(
		*specialsJson.Data,
		&specialTokens,
	); specialErr != nil {
		return nil, specialErr
	}

	for k, v := range specialTokens {
		var specialToken string
		switch t := v.(type) {
		case string:
			specialToken = t
		case JsonMap:
			mv := t["content"]
			switch mvt := mv.(type) {
			case string:
				specialToken = mvt
			default:
				log.Fatalf(
					"unknown format for `special_tokens_map."+
						"json`: %v", t,
				)
			}
		default:
			log.Fatalf(
				"unknown format for `special_tokens_map."+
					"json`: %v", t,
			)
		}
		realizedSpecials[k] = specialToken
	}
	if len(realizedSpecials) > 0 {
		specialsFile, specialFileErr := os.OpenFile(
			path.Join(dir, "specials.json"),
			os.O_TRUNC|os.O_RDWR|os.O_CREATE, 0755,
		)
		if specialFileErr != nil {
			return nil, fmt.Errorf(
				"cannot generate specials.json: %s",
				specialFileErr,
			)
		}
		specialsJsonBytes, specialsErr := json.Marshal(realizedSpecials)
		if specialsErr != nil {
			_ = specialsFile.Close()
			return nil, fmt.Errorf(
				"cannot marshal specials.json: %s", specialsErr,
			)
		}
		if _, writeErr := specialsFile.Write(
			specialsJsonBytes,
		); writeErr != nil {
			_ = specialsFile.Close()
			return nil, fmt.Errorf(
				"cannot write specials.json: %s", specialsErr,
			)
		}
		if _, seekErr := specialsFile.Seek(0, 0); seekErr != nil {
			return nil,
				fmt.Errorf("cannot seek specials.json: %s", seekErr)
		}
		if mmapErr := rsrcs.AddEntry(
			"specials.json",
			specialsFile,
		); mmapErr != nil {
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
	foundResources := make(Resources)
	resources := GetResourceEntries(rsrcType)
	aliases := getResourceEntryAliases(rsrcType)

	for file, flag := range resources {
		var rsrcFile os.File

		// Resolve the resource
		if flag <= rsrcLvl {
			log.Printf("Resolving %s/%s... ", uri, file)
			targetPath := path.Join(*dir, file)
			rsrcSize, rsrcSizeErr := Size(uri, file, token)
			alias := file
			if rsrcSizeErr != nil {
				// If the resource isn't found under its normal filename,
				// check under any known aliases.
				if aliasesList, ok := aliases[file]; ok {
					for _, alias = range aliasesList {
						rsrcSize, rsrcSizeErr = Size(uri, alias, token)
						if rsrcSizeErr == nil {
							log.Printf(
								"Resolving %s/%s as alias %s/%s...",
								uri, file, uri, alias,
							)
							break
						}
					}
				}
			}
			if rsrcSizeErr != nil {
				// If the resource is required, we cannot continue.
				if flag&RESOURCE_REQUIRED != 0 {
					log.Printf(
						"%s/%s not found, required!",
						uri, file,
					)
					return &foundResources, fmt.Errorf(
						"cannot retrieve required `%s from %s`: %s",
						uri, file, rsrcSizeErr,
					)
				} else {
					// Otherwise, we can skip it.
					continue
				}
				// If the resource exists, and is the correct size, skip it.
			} else if targetStat, targetStatErr := os.Stat(targetPath); !os.IsNotExist(
				targetStatErr,
			) && uint(targetStat.Size()) == rsrcSize {
				log.Printf(
					"Skipping %s/%s... already exists, "+
						"and of the correct size.", uri, file,
				)
				openFile, skipFileErr := os.OpenFile(
					path.Join(*dir, file),
					os.O_RDONLY, 0755,
				)
				if skipFileErr != nil {
					return &foundResources, fmt.Errorf(
						"error opening '%s' for read: %s",
						file, skipFileErr,
					)

				} else {
					// If the resource exists, but is the wrong size, we need
					// to download it.
					rsrcFile = *openFile
				}
			} else if rsrcReader, rsrcErr := Fetch(
				uri, alias, token,
			); rsrcErr != nil {
				return &foundResources, fmt.Errorf(
					"cannot retrieve `%s from %s`: %s",
					uri, alias, rsrcErr,
				)
			} else {
				if dirErr := os.MkdirAll(
					path.Dir(path.Join(*dir, file)), 0755,
				); dirErr != nil {
					return &foundResources, fmt.Errorf(
						"cannot create directory for '%s': %s",
						file, dirErr,
					)
				}
				openFile, rsrcFileErr := os.OpenFile(
					path.Join(*dir, file),
					os.O_TRUNC|os.O_RDWR|os.O_CREATE, 0755,
				)
				if rsrcFileErr != nil {
					return &foundResources, fmt.Errorf(
						"error opening '%s' for write: %s",
						file, rsrcFileErr,
					)
				}
				rsrcFile = *openFile

				counter := &WriteCounter{
					Last: time.Now(),
					Path: fmt.Sprintf("%s/%s", uri, file),
					Size: uint64(rsrcSize),
				}
				bytesDownloaded, ioErr := io.Copy(
					&rsrcFile,
					io.TeeReader(rsrcReader, counter),
				)
				_ = rsrcReader.Close()
				if ioErr != nil {
					return &foundResources, fmt.Errorf(
						"error downloading '%s': %s",
						alias, ioErr,
					)
				} else {
					log.Printf(
						"Downloaded %s/%s... "+
							"%s completed.", uri, alias,
						humanize.Bytes(uint64(bytesDownloaded)),
					)
				}
			}

			if mmapErr := foundResources.AddEntry(
				file,
				&rsrcFile,
			); mmapErr != nil {
				return &foundResources, fmt.Errorf(
					"error trying to mmap file: %s",
					mmapErr,
				)
			}
		}
	}

	// check if tokenizer.model exists, if so, expand to files
	flagTokenizerModelExist := CheckFileExist(
		path.Join(
			*dir, "tokenizer.model",
		),
	)
	if flagTokenizerModelExist {
		// check size of tokenizer.model
		targetStat, targetStatErr := os.Stat(
			path.Join(
				*dir, "tokenizer.model",
			),
		)
		if targetStatErr != nil {
			return &foundResources, fmt.Errorf(
				"cannot stat tokenizer.model: %s",
				targetStatErr,
			)
		}
		if targetStat.Size() == 0 {
			flagTokenizerModelExist = false
		}
	}

	if flagTokenizerModelExist {
		log.Printf(
			"Directory %s contains tokenizer.model, extracting to files",
			path.Join(*dir, "tokenizer.model"),
		)
		ConvertSentencepieceFiles(
			path.Join(*dir, "tokenizer.model"),
			false,
		)

		// Add the new files to the resources
		files, _ := os.ReadDir(*dir)
		for _, f := range files {
			// If not already in the resources, add it
			if _, ok := foundResources[f.Name()]; !ok {
				openFile, rsrcFileErr := os.OpenFile(
					path.Join(*dir, f.Name()),
					os.O_RDONLY, 0755,
				)
				if rsrcFileErr != nil {
					return &foundResources, fmt.Errorf(
						"error opening '%s' for read: %s",
						f.Name(), rsrcFileErr,
					)
				}
				rsrcFile := *openFile
				if mmapErr := foundResources.AddEntry(
					f.Name(),
					&rsrcFile,
				); mmapErr != nil {
					return &foundResources, fmt.Errorf(
						"error trying to mmap file: %s",
						mmapErr,
					)
				}
				log.Printf(
					"Added %s to resources via sentencepiece conversion\n",
					f.Name(),
				)
			}
		}

	} else {
		// check if tokenizer exists by checking if tokenizer.json exists
		// and has data in it
		flagTokenizerExist := CheckFileExist(
			path.Join(
				*dir, "tokenizer.json",
			),
		)
		if flagTokenizerExist {
			// check size of tokenizer.json
			targetStat, targetStatErr := os.Stat(
				path.Join(
					*dir, "tokenizer.json",
				),
			)
			if targetStatErr != nil {
				return &foundResources, fmt.Errorf(
					"cannot stat tokenizer.json: %s",
					targetStatErr,
				)
			}
			if targetStat.Size() == 0 {
				flagTokenizerExist = false
			}
		}

		// if tokenizer exists, but vocab and merges do not exist, extract
		// from tokenizer, else if vocab and merges exist, do nothing; if
		// both do not exist, fail
		flagVocabExist := CheckFileExist(path.Join(*dir, "vocab.json"))
		flagMergesExists := CheckFileExist(path.Join(*dir, "merges.txt"))

		if flagTokenizerExist {
			// if vocab does not exist, extract it from tokenizer
			if !flagVocabExist {
				model, err := ExtractModelFromTokenizer(dir)
				if err != nil {
					return &foundResources, fmt.Errorf(
						"could not extract model from tokenizer %s",
						err,
					)
				}

				err = ExtractVocabFromTokenizer(model, dir, &foundResources)
				if err != nil {
					return &foundResources, fmt.Errorf(
						"could not extract vocab from tokenizer %s",
						err,
					)
				}
			}

			// if merges does not exist, extract it from tokenizer
			if !flagMergesExists {
				model, err := ExtractModelFromTokenizer(dir)
				if err != nil {
					return &foundResources,
						fmt.Errorf(
							"could not extract model from tokenizer %s",
							err,
						)
				}

				err = ExtractMergesFromTokenizer(model, dir, &foundResources)
				if err != nil {
					return &foundResources, fmt.Errorf(
						"could not extract merges from tokenizer %s",
						err,
					)
				}
			}
		} else {
			// if tokenizer does not exist, check if vocab and merges exist
			if flagVocabExist && flagMergesExists {
				// if both exist, do nothing
				log.Println(
					"Vocab and merges exist, but tokenizer does not. OK",
				)
			} else {
				// if either does not exist, fail
				return &foundResources, fmt.Errorf(
					"tokenizer, vocab, and merges do not exist",
				)
			}
		}

	}

	// Check if we already got the pytorch model file
	flagModelExists := CheckFileExist(path.Join(*dir, "pytorch_model.bin"))
	log.Printf("Pytorch Model File exists: %t\n", flagModelExists)

	// if model does not exist, check if we have the sharded config
	if !flagModelExists {
		flagShardConfigExists := CheckFileExist(
			path.Join(
				*dir, "pytorch_model.bin.index.json",
			),
		)
		log.Printf("Shard config exists: %t", flagShardConfigExists)
		//if sharded config exists, attempt to download the shards
		if flagShardConfigExists {
			numShards, err := FindNumberOfShardsFromConfig(
				path.Join(
					*dir, "pytorch_model.bin.index.json",
				),
			)
			if err != nil {
				log.Printf(
					"Could not find number of shards from config: %s\n",
					err,
				)
				return &foundResources, errors.New(
					"could not find number of shards from config",
				)
			}

			// pad the number of shards to 5 digits
			log.Printf("Found %d shards\n", numShards)
			paddedTotalShards := fmt.Sprintf("%05d", numShards)

			// loop through shards and download them
			for i := 1; i <= numShards; i++ {
				var rsrcFile os.File

				paddedShardString := fmt.Sprintf("%05d", i)
				// Construct the shard path
				shardPath := fmt.Sprintf(
					"pytorch_model-%s-of-%s.bin", paddedShardString,
					paddedTotalShards,
				)
				log.Printf("Resolving shard %s\n", shardPath)

				targetPath := path.Join(*dir, shardPath)
				rsrcSize, rsrcSizeErr := Size(uri, shardPath, token)
				if rsrcSizeErr != nil {
					fmt.Printf(
						"could not get size of shard %s: %s\n",
						shardPath,
						rsrcSizeErr,
					)
					return &foundResources,
						errors.New("could not get size of shard")
				}
				// Print size of shard
				log.Printf(
					"Remote size of shard %s is %s\n", shardPath,
					humanize.Bytes(uint64(rsrcSize)),
				)

				// Check if shard exists locally
				flagShardExists := CheckFileExist(targetPath)
				if flagShardExists {
					// Check if shard local size is correct compared to
					// remote size
					localShardInfo, err := os.Stat(targetPath)
					if err != nil {
						fmt.Printf(
							"Could not get size of local shard %s: %s\n",
							shardPath, err,
						)
						return &foundResources,
							errors.New("could not get size of local shard")
					}
					if (rsrcSize > 0) && (rsrcSize == uint(localShardInfo.Size())) {
						log.Printf(
							"Skipping shard %s, exists and correct size\n",
							shardPath,
						)
						continue
					}
				}

				// fetch shard
				var rsrcReader io.ReadCloser
				rsrcReader, err = Fetch(uri, shardPath, token)
				if err != nil {
					return &foundResources, fmt.Errorf(
						"error trying to fetch file: %s", err,
					)
				}

				// create shard file
				var rsrcFilePtr *os.File
				rsrcFilePtr, err = os.Create(targetPath)
				if err != nil {
					return &foundResources, fmt.Errorf(
						"error trying to create file: %s", err,
					)
				}
				rsrcFile = *rsrcFilePtr

				// copy shard to file
				counter := &WriteCounter{
					Last: time.Now(),
					Path: fmt.Sprintf("%s/%s", uri, shardPath),
					Size: uint64(rsrcSize),
				}
				bytesDownloaded, ioErr := io.Copy(
					&rsrcFile,
					io.TeeReader(rsrcReader, counter),
				)
				//close shard reader
				err = rsrcReader.Close()

				if err != nil {
					return &foundResources, fmt.Errorf(
						"error trying to close reader: %s", err,
					)
				}
				if ioErr != nil {
					return &foundResources, fmt.Errorf(
						"error downloading '%s': %s",
						shardPath,
						ioErr,
					)
				} else {
					log.Printf(
						"Downloaded %s/%s... "+
							"%s completed.", uri, shardPath,
						humanize.Bytes(uint64(bytesDownloaded)),
					)
				}

				//close shard file
				err = rsrcFile.Close()

				if err != nil {
					return &foundResources, fmt.Errorf(
						"error trying to close reader: %s", err,
					)
				} else {
					log.Printf(
						"Downloaded %s/%s... "+
							"%s completed.", uri, shardPath,
						humanize.Bytes(uint64(bytesDownloaded)),
					)
				}

				//check if shard local size is correct compared to remote size
				var localShardInfo os.FileInfo
				localShardInfo, err = os.Stat(targetPath)
				if err != nil {
					fmt.Printf(
						"Could not get size of local shard %s: %s\n",
						shardPath, err,
					)
					return &foundResources,
						errors.New("could not get size of local shard")
				}
				if (rsrcSize > 0) &&
					(rsrcSize != uint(localShardInfo.Size())) {
					return &foundResources,
						errors.New("shard was not downloaded correctly")
				}
				log.Printf(
					"Shard %s downloaded correctly, size is %s\n",
					shardPath,
					humanize.Bytes(uint64(localShardInfo.Size())),
				)

			}
			log.Printf("Downloaded %d shards\n", numShards)

		}

	}
	return &foundResources, nil
}

// ResolveSplitRegex
// Resolves the split regex from the tokenizer.json file, if it exists.
func (rsrcs *Resources) ResolveSplitRegex() *string {
	var splitRegex *string
	if tokenizerData, ok := (*rsrcs)["tokenizer.json"]; ok {
		var tokenizerMap JsonMap
		if tokenizerData.Data != nil {
			if json.Unmarshal(*tokenizerData.Data, &tokenizerMap) != nil {
				log.Fatal("Error unmarshalling tokenizer.json")
			}
		}
		if preTok, ok := tokenizerMap["pre_tokenizer"]; ok && preTok != nil {
			preTokMap := preTok.(map[string]interface{})
			if pretokenizers, ok := preTokMap["pretokenizers"]; ok &&
				pretokenizers != nil {
				pretokenizersList := pretokenizers.([]interface{})
				for _, v := range pretokenizersList {
					vIntf := v.(map[string]interface{})
					if vIntf["type"] == "Split" {
						pattern := vIntf["pattern"].(map[string]interface{})
						if pattern["Regex"] != nil {
							splitRegexVal := pattern["Regex"].(string)
							// Fix lookbacks
							splitRegexVal = strings.ReplaceAll(
								splitRegexVal,
								"(?!\\S)",
								"(\\S){0}",
							)
							splitRegex = &splitRegexVal
						}
					}
				}
			}
		}
	}
	return splitRegex
}

// CheckFileExist checks if a file exists at a given path.
func CheckFileExist(path string) bool {
	_, err := os.Stat(path)

	if errors.Is(err, os.ErrNotExist) {
		return false
	} else {
		return true
	}
}

// HFConfig contains the tokenizer configuration that gpt_bpe uses.
type HFConfig struct {
	ModelId             *string   `json:"omitempty"`
	ModelType           *string   `json:"model_type,omitempty"`
	EosTokenId          *Token    `json:"eos_token_id,omitempty"`
	BosTokenId          *Token    `json:"bos_token_id,omitempty"`
	PadTokenId          *Token    `json:"pad_token_id,omitempty"`
	BosTokenStr         *string   `json:"bos_token,omitempty"`
	EosTokenStr         *string   `json:"eos_token,omitempty"`
	PadTokenStr         *string   `json:"pad_token,omitempty"`
	VocabSize           *uint32   `json:"vocab_size,omitempty"`
	NewLineMode         *string   `json:"newlinemode,omitempty"`
	TokenizerClass      *string   `json:"tokenizer_class"`
	AddBosToken         *bool     `json:"add_bos_token,omitempty"`
	AddEosToken         *bool     `json:"add_eos_token,omitempty"`
	AddedSpecialsTokens *TokenMap `json:"added_specials_tokens,omitempty"`
	IgnoreMerges        *bool     `json:"ignore_merges,omitempty"`
}

// SpecialConfig contains the special tokens and special token configuration
// that gpt_bpe uses.
type SpecialConfig struct {
	PuncRunes     []*string          `json:"punc_runes"`
	Normalizer    *map[string]string `json:"normalizer"`
	EncloseEosBos bool               `json:"enclose_eos_bos"`
	PrefixSpace   bool               `json:"prefix_space"`
	LowerCase     bool               `json:"lower_case"`
	EndOfWord     string             `json:"end_of_word"`
	DecodeExtra   *map[string]string `json:"decode_extra"`
	SplitRegex    *string            `json:"split_regex"`
}

// NewHFConfig creates a new HFConfig object with default values.
func NewHFConfig() *HFConfig {
	defaultModelId := ""
	defaultModelType := "gpt2"
	defaultEosTokenId := Token(0)
	defaultBosTokenId := Token(0)
	defaultPadTokenId := Token(0)
	defaultBosTokenStr := "<|startoftext|>"
	defaultEosTokenStr := "<|endoftext|>"
	defaultPadTokenStr := ""
	defaultVocabSize := uint32(50257)
	defaultNewLineMode := "prefix"
	defaultTokenizerClass := "GPT2BPETokenizer"
	defaultAddBosToken := false
	defaultAddEosToken := false
	defaultAddedSpecialsTokens := make(TokenMap)
	HFConfig := &HFConfig{
		ModelId:             &defaultModelId,
		ModelType:           &defaultModelType,
		EosTokenId:          &defaultEosTokenId,
		BosTokenId:          &defaultBosTokenId,
		PadTokenId:          &defaultPadTokenId,
		BosTokenStr:         &defaultBosTokenStr,
		EosTokenStr:         &defaultEosTokenStr,
		PadTokenStr:         &defaultPadTokenStr,
		VocabSize:           &defaultVocabSize,
		NewLineMode:         &defaultNewLineMode,
		TokenizerClass:      &defaultTokenizerClass,
		AddBosToken:         &defaultAddBosToken,
		AddEosToken:         &defaultAddEosToken,
		AddedSpecialsTokens: &defaultAddedSpecialsTokens,
	}
	return HFConfig
}

// Processor stores config to process one step of the pipeline
type Processor struct {
	ProcessorType string
	ProcessorArgs JsonMap
}

// Process the input with the processor
func (p *Processor) Process(input interface{}) (interface{}, error) {
	switch p.ProcessorType {
	case "prepend":
		return nil, errors.New("prepend not implemented")
	default:
		return nil, errors.New("unknown processor type")
	}
}

// LoadExternalResources
// Resolves a given vocabulary id, and returns the corresponding HuggingFace
// configuration, and the resources for the tokenizer.
func LoadExternalResources(
	vocabId string,
	token string,
) (resources *Resources, err error) {
	dir, dirErr := ioutil.TempDir("", "resources")
	if dirErr != nil {
		return nil, dirErr
	}
	defer func(path string) {
		_ = os.RemoveAll(path)
	}(dir)
	rslvdResources, rsrcErr := ResolveResources(
		vocabId,
		&dir,
		RESOURCE_DERIVED,
		RESOURCETYPE_TRANSFORMERS,
		token,
	)
	if rsrcErr != nil {
		return nil, rsrcErr
	} else {
		resources = rslvdResources
	}
	fmt.Printf("Resources: %v\n", resources)
	return resources, nil

}

// ResolveHF
// Given a set of resources, resolve the HuggingFace configuration.
// Used to be able to resolve both embedded and local resources.
func (rsrcs *Resources) ResolveHF(hfConfig *HFConfig) (err error) {
	// Resolve config and tokenizer config from resources
	// config.json and tokenizer_config.json
	if err = rsrcs.resolveConfigAndTokenizer(hfConfig); err != nil {
		return err
	}

	// Resolve special tokens and special tokens config from resources
	// special_tokens_map.json and specials.txt
	if err = rsrcs.resolveSpecials(hfConfig); err != nil {
		return err
	}

	// Resolve Vocab size from vocab.json or encoder.json
	if err = rsrcs.resolveVocabSize(hfConfig); err != nil {
		return err
	}

	// Sometimes TokenIDs are not properly resolved, so we need to check
	if hfConfig != nil {
		if *hfConfig.EosTokenId == 0 || *hfConfig.BosTokenId == 0 ||
			*hfConfig.PadTokenId == 0 {
			if err = rsrcs.resolveTokenIds(hfConfig); err != nil {
				return err
			}
		}
	} else {
		return errors.New("could not resolve HFConfig")
	}

	// Llama 3 and other larger models will enclose eos and bos by default
	if *hfConfig.VocabSize > math.MaxUint16+1 {
		var addEosToken = true
		var addBosToken = true

		hfConfig.AddEosToken = &addEosToken
		hfConfig.AddBosToken = &addBosToken
	}

	return nil
}

func GetMergesAsBpeRank(resources *Resources) (map[GPTPair]float64, error) {
	bpeRanks := make(map[GPTPair]float64)
	// Try to get from merges.txt
	if mergesTxt, ok := (*resources)["merges.txt"]; ok {
		scanner := bufio.NewScanner(bytes.NewBuffer(*mergesTxt.Data))
		idx := uint32(0)
		firstLine := true
		for scanner.Scan() {
			if firstLine {
				firstLine = false
				continue
			}
			leftRight := strings.SplitN(scanner.Text(), " ", 2)
			bpeRanks[GPTPair{
				Left:  leftRight[0],
				Right: leftRight[1],
			}] = float64(idx)
			idx += 1
		}
	} else if mergesJson, ok := (*resources)["merges.json"]; ok {
		var mergesTable [][]string
		err := json.Unmarshal(*mergesJson.Data, &mergesTable)
		if err != nil {
			return nil, err
		}
		// Iterate over the merges and add them to the BPE ranks
		for rank, merge := range mergesTable {
			bpeRanks[GPTPair{merge[0], merge[1]}] =
				float64(rank)
		}
	} else if tokenizerJson, ok := (*resources)["tokenizer.json"]; ok {
		// Finally try to get from tokenizer.json, merges entry
		var tokenizerJsonMap JsonMap
		err := json.Unmarshal(*tokenizerJson.Data, &tokenizerJsonMap)
		if err != nil {
			return nil, err
		}

		model, ok := tokenizerJsonMap["model"]
		if !ok {
			return nil, errors.New("could not get model from tokenizer.json")
		}
		merges, ok := model.(map[string]interface{})["merges"]
		if !ok {
			return nil, errors.New("could not get merges from tokenizer.json")
		}
		// Iterate over the merges and add them to the BPE ranks, in form of string[]
		for rank, merge := range merges.([]interface{}) {
			mergeStr := merge.(string)
			mergeSplit := strings.Split(mergeStr, " ")
			bpeRanks[GPTPair{mergeSplit[0], mergeSplit[1]}] =
				float64(rank)
		}
	} else {
		return nil, errors.New("could not find merges")
	}
	return bpeRanks, nil
}

func (rsrcs *Resources) UnmarshalData(
	name string,
) (data *JsonMap, err error) {
	if _, err = (*rsrcs).GetFile(name); err == nil {
		rawData := (*rsrcs)[name].Data
		if err = json.Unmarshal(*rawData, &data); err != nil {
			return nil,
				fmt.Errorf("error unmarshalling %s: %s", name, err)
		}
	} else {
		return nil, nil
	}
	return data, nil
}

func (rsrcs *Resources) UnmarshalUntilData(
	names []string,
) (
	name string,
	data *JsonMap,
	err error,
) {
	for _, name = range names {
		if _, ok := (*rsrcs)[name]; !ok {
			continue
		}
		if data, err = rsrcs.UnmarshalData(name); err == nil && data != nil {
			return name, data, nil
		} else if err != nil {
			return name, nil, fmt.Errorf(
				"error unmarshalling %s: %s", name, err,
			)
		}
	}
	return "", nil, nil
}

// GetVocab
// Get the vocab from the resources.
// Adapter function to get the vocab from either vocab.json or encoder.json.
func (rsrcs *Resources) GetVocab(
	hfConfig *HFConfig,
) (TokenMap, error) {
	// Vocab is stored in either the vocab.json or encoder.json file
	// We want to unmarshal the vocab file into an interface to work with
	// We attempt to unmarshal under the vocab.json key first, then
	// encoder.json if it fails
	filesToAttempt := []string{"vocab.json", "encoder.json", "tokenizer.json"}

	// Get the vocab from the resources
	name, vocabData, err := rsrcs.UnmarshalUntilData(filesToAttempt)
	if err != nil {
		return nil, err
	} else if vocabData == nil {
		return nil, errors.New("vocab file not found")
	}

	tokens := make(TokenMap)
	if name == "tokenizer.json" {
		// We get the vocab stored in the /model/vocab key
		if modelInterface, ok := (*vocabData)["model"]; ok {
			model := modelInterface.(map[string]interface{})
			if vocabInterface, ok := model["vocab"]; ok {
				vocabMap := vocabInterface.(map[string]interface{})
				for k, v := range vocabMap {
					tokens[k] = types.Token(v.(float64))
				}
			}
			// Check for "ignore_merges", and set it
			if ignoreMerges, ok := model["ignore_merges"].(bool); ok {
				hfConfig.IgnoreMerges = &ignoreMerges
			}
		}
	} else {
		for k, v := range *vocabData {
			tokens[k] = types.Token(v.(float64))
		}
	}

	if hfConfig.IgnoreMerges == nil {
		disableIgnoreMerges := false
		hfConfig.IgnoreMerges = &disableIgnoreMerges
	}

	// Add the special tokens to the vocab
	if len(*hfConfig.AddedSpecialsTokens) > 0 {
		for k, v := range *hfConfig.AddedSpecialsTokens {
			tokens[k] = v
		}
	}

	return tokens, nil
}

// resolveTokenIds
// Resolve token ids for eos, bos, and pad tokens from resources.
func (rsrcs *Resources) resolveTokenIds(hfConfig *HFConfig) error {
	// Get the vocab from the resources
	vocab, err := rsrcs.GetVocab(hfConfig)
	if err != nil {
		return err
	}

	// Get the token ids for eos, bos, and pad tokens
	var eosTokenId, bosTokenId, padTokenId *Token
	if eosToken, ok := vocab[*hfConfig.EosTokenStr]; ok {
		eosTokenId = new(Token)
		*eosTokenId = Token(eosToken)
		hfConfig.EosTokenId = eosTokenId
	}
	if bosToken, ok := vocab[*hfConfig.BosTokenStr]; ok {
		bosTokenId = new(Token)
		*bosTokenId = Token(bosToken)
		hfConfig.BosTokenId = bosTokenId
	}
	if padToken, ok := vocab[*hfConfig.PadTokenStr]; ok {
		padTokenId = new(Token)
		*padTokenId = Token(padToken)
		hfConfig.PadTokenId = padTokenId
	}

	return nil
}

// resolveVocabSize
// Resolve vocab size from resources.
// Used to be able to resolve both embedded and local resources.
// Continuation of ResolveHFFromResources.
func (rsrcs *Resources) resolveVocabSize(hfConfig *HFConfig) (err error) {
	// Get the vocab from the resources
	var vocab TokenMap
	if vocab, err = rsrcs.GetVocab(hfConfig); err != nil {
		return err
	}

	// Get length of vocab
	vocabLen := new(uint32)
	*vocabLen = uint32(len(vocab))

	hfConfig.VocabSize = vocabLen
	return nil
}

// resolveConfigAndTokenizer
// Resolve config and tokenizer config from resources.
// Used to be able to resolve both embedded and local resources.
// Continuation of ResolveHFFromResources.
func (rsrcs *Resources) resolveConfigAndTokenizer(
	hfConfig *HFConfig,
) (err error) {
	// Use interfaces to unmarshal the config file and tokenizer config file
	var config *JsonMap
	var tokenizerConfig *JsonMap

	// Get the config and tokenizer config from the resources

	// If exists, unmarshal config.json and tokenizer_config.json, else
	// use GetFile to get the file, then unmarshal it

	if config, err = rsrcs.UnmarshalData("config.json"); err != nil {
		return err
	}
	if tokenizerConfig, err =
		rsrcs.UnmarshalData("tokenizer_config.json"); err != nil {
		return err
	}

	// Check if bos_token is in string, this is the old format Pythia has.
	// If not, try to unmarshal to the tokenizerSpecials
	// that llama 2 has, else try mistral format
	if config != nil || tokenizerConfig != nil {
		hasReadForEosBos := false

		// Read config.json
		if config != nil {
			configMap := *config
			// Using interfaces, first check if bos_token is in string format
			if bosToken, ok := configMap["bos_token"].(string); ok {
				hfConfig.BosTokenStr = &bosToken
				if eosToken, ok := configMap["eos_token"].(string); ok {
					hfConfig.EosTokenStr = &eosToken
				}
				if padToken, ok := configMap["pad_token"].(string); ok {
					hfConfig.PadTokenStr = &padToken
				}
				hasReadForEosBos = true
			}

			// Read for EOS BOS token ID
			if eosTokenId, ok := configMap["eos_token_id"].(float64); ok {
				eosTokenIdInt := Token(eosTokenId)
				hfConfig.EosTokenId = &eosTokenIdInt
			}
			if bosTokenId, ok := configMap["bos_token_id"].(float64); ok {
				bosTokenIdInt := Token(bosTokenId)
				hfConfig.BosTokenId = &bosTokenIdInt
			}

			// Read for vocab size
			if vocabSize, ok := configMap["vocab_size"].(float64); ok {
				vocabSizeInt := uint32(vocabSize)
				hfConfig.VocabSize = &vocabSizeInt
			}

			// Read for newLineMode
			if newLineMode, ok := configMap["newlinemode"].(string); ok {
				hfConfig.NewLineMode = &newLineMode
			}
		}

		// Read tokenizer_config.json
		if tokenizerConfig != nil {
			configMap := *tokenizerConfig
			if !hasReadForEosBos {
				// Using interfaces, first check if bos_token is in string format
				if bosToken, ok := configMap["bos_token"].(string); ok {
					hfConfig.BosTokenStr = &bosToken
					if eosToken, ok := configMap["eos_token"].(string); ok {
						hfConfig.EosTokenStr = &eosToken
					}
					if padToken, ok := configMap["pad_token"].(string); ok {
						hfConfig.PadTokenStr = &padToken
					}
					hasReadForEosBos = true

				}
			}
			// If not, assume llama2 format and try to unmarshal
			if !hasReadForEosBos {
				if bosToken, ok :=
					configMap["bos_token"].(map[string]interface{}); ok {
					if content, ok := bosToken["content"].(string); ok {
						hfConfig.BosTokenStr = &content
					}
				}
				if eosToken, ok :=
					configMap["eos_token"].(map[string]interface{}); ok {
					if content, ok := eosToken["content"].(string); ok {
						hfConfig.EosTokenStr = &content
					}
				}
				if padToken, ok := configMap["pad_token"].(string); ok {
					hfConfig.PadTokenStr = &padToken
				}
			}
			// If that doesn't work, assume mistral format
			if !hasReadForEosBos {
				if bosToken, ok := configMap["bos_token"].(string); ok {
					hfConfig.BosTokenStr = &bosToken
				}
				if eosToken, ok := configMap["eos_token"].(string); ok {
					hfConfig.EosTokenStr = &eosToken
				}
				if padToken, ok := configMap["pad_token"].(string); ok {
					hfConfig.PadTokenStr = &padToken
				}
			}

			// Read for enclose eos bos
			if encloseEos, ok := configMap["add_bos_token"].(bool); ok {
				hfConfig.AddBosToken = &encloseEos
			}

			if encloseBos, ok := configMap["add_eos_token"].(bool); ok {
				hfConfig.AddEosToken = &encloseBos
			}

			// Read for added_specials_tokens
			// Will later be used to readd into vocab if needed
			if addedTokensDecoder, ok :=
				configMap["added_tokens_decoder"].(map[string]interface{}); ok {
				addedSpecialsTokens := make(TokenMap)
				for k, v := range addedTokensDecoder {
					// Get under content key, key is float64
					keyToken, _ := strconv.ParseFloat(k, 64)
					valStr := v.(map[string]interface{})["content"].(string)
					addedSpecialsTokens[valStr] = types.Token(keyToken)
				}
				hfConfig.AddedSpecialsTokens = &addedSpecialsTokens
			}

			// Read for tokenizer Class
			if tClass, ok := configMap["tokenizer_class"].(string); ok {
				hfConfig.TokenizerClass = &tClass
			}
		}
	}
	return nil
}

// resolveSpecials
// Resolve special tokens and special config from resources.
// Used to be able to resolve both embedded and local resources.
// Continuation of ResolveHFFromResources.
func (rsrcs *Resources) resolveSpecials(hfConfig *HFConfig) error {
	// Get specials config from resources
	// We can only generate specials.json if we have special_tokens_map
	specialsJson, ok := (*rsrcs)["special_tokens_map.json"]
	if ok {
		specialTokens := make(JsonMap)
		if specialErr := json.Unmarshal(
			*specialsJson.Data,
			&specialTokens,
		); specialErr != nil {
			return specialErr
		}

		// Try to get pad token from specials if not already set
		if hfConfig.PadTokenStr == nil || *hfConfig.PadTokenStr == "" {
			if padToken, pOk := specialTokens["pad_token"].(string); pOk {
				hfConfig.PadTokenStr = &padToken
			}
		}
	}

	// Get from specials.json
	specialsTxt, ok := (*rsrcs)["specials.txt"]
	if ok {
		// Treat specials.txt as an array of strings and try to match
		specials := strings.Split(string(*specialsTxt.Data), "\n")
		if hfConfig.PadTokenStr == nil {
			for _, special := range specials {
				if strings.Contains(strings.ToLower(special), "pad") {
					hfConfig.PadTokenStr = &special
					break
				}
			}
		}
	}
	return nil
}

func (rsrcs *Resources) LoadEmbeddedResource(
	vocabId string,
	resourceId string,
	path string,
) {
	if r := GetEmbeddedResource(vocabId + "/" + path); r != nil {
		(*rsrcs)[resourceId] = *r
	}
}

// ResolveResourcesList
// Resolves a list of resources, and checks if they exist in the given
// directory. If they don't exist, they are downloaded.
func ResolveResourcesList(vocabId string, token string) (*Resources, error) {
	// Resolve the vocab id - Embedded resources
	if _, vocabErr := EmbeddedDirExists(vocabId); vocabErr == nil {
		resources := make(Resources)

		possibleEmbeddedResources := []struct {
			resourceId string
			path       string
		}{
			{"vocab.json", "encoder.json"},
			{"config.json", "config.json"},
			{"merges.txt", "vocab.bpe"},
			{"merges.json", "merges.json"},
			{"specials.txt", "specials.txt"},
			{"special_tokens_map.json", "special_tokens_map.json"},
			{"special_config.json", "special_config.json"},
			{"tokenizer.json", "tokenizer.json"},
			{"tokenizer_config.json", "tokenizer_config.json"},
		}

		for _, resource := range possibleEmbeddedResources {
			resources.LoadEmbeddedResource(
				vocabId, resource.resourceId, resource.path,
			)
		}
		return &resources, nil
	}
	// Non-embedded resources
	resources, err := LoadExternalResources(vocabId, token)
	if err != nil {
		return nil, err
	}
	return resources, nil

}

// ResolveVocabId
// Resolves a vocabulary id to a set of resources, from embedded,
// local filesystem, or remote, and applies processing to the resources.
func ResolveVocabId(vocabId string, token string) (
	*HFConfig,
	*Resources,
	error,
) {
	rsrcs, err := ResolveResourcesList(vocabId, token)
	if err != nil {
		return nil, nil, err
	}

	hf := NewHFConfig()
	hf.ModelId = &vocabId
	if err = rsrcs.ResolveHF(hf); err != nil {
		return nil, nil, err
	}
	return hf, rsrcs, nil
}

func ExtractModelFromTokenizer(dir *string) (JsonMap, error) {
	tokenizerPath := path.Join(*dir, "tokenizer.json")
	// Open the file
	tokenizerFile, err := os.Open(tokenizerPath)
	if err != nil {
		log.Println("Error opening tokenizer:", err)
		// return an empty map and the error
		return nil, err
	}
	defer func(tokenizerFile *os.File) {
		_ = tokenizerFile.Close()
	}(tokenizerFile)

	// Decode the JSON data into a map
	var data JsonMap
	err = json.NewDecoder(tokenizerFile).Decode(&data)
	if err != nil {
		log.Println("Error decoding JSON from tokenizer:", err)
		return nil, err
	}

	// Access the data at the specified path
	model, ok := (data["model"]).(map[string]interface{})
	model = ToJsonMap(model)
	if ok {
		return model, nil
	} else {
		log.Println("Error: Could not convert model in tokenizer to map")
		return nil, errors.New("could not convert model in tokenizer to map")
	}
}

func ExtractVocabFromTokenizer(
	model JsonMap,
	dir *string,
	resources *Resources,
) error {
	vocab, ok := model["vocab"].(map[string]interface{})
	vocab = ToJsonMap(vocab)
	if !ok {
		log.Println("Error: Could not convert vocab in model to map")
		return errors.New("could not convert vocab in model to map")
	}

	vocabPath := path.Join(*dir, "vocab.json")

	// Create the file
	vocabFile, err := os.Create(vocabPath)
	if err != nil {
		log.Println("Error creating vocab.json:", err)
		return err
	}
	defer func(vocabFile *os.File) {
		_ = vocabFile.Close()
	}(vocabFile)

	// Marshal the vocab map into a JSON string with indentation
	vocabJsonString, err := json.MarshalIndent(vocab, "", " ")
	if err != nil {
		fmt.Println("Error marshaling JSON:", err)
		return err
	}

	// Write the JSON string to the file
	_, err = vocabFile.Write(vocabJsonString)
	if err != nil {
		log.Println("Error writing to vocab.json:", err)
		return err
	}

	log.Println("Vocab written to vocab.json from tokenizer.json")

	if mmapErr := resources.AddEntry(
		"vocab.json", vocabFile,
	); mmapErr != nil {
		return fmt.Errorf("error trying to mmap file: %s", mmapErr)
	}

	return nil
}

func ExtractMergesFromTokenizer(
	model JsonMap,
	dir *string,
	resources *Resources,
) error {
	merges, ok := model["merges"].([]interface{})
	if !ok {
		log.Println("Error: Could not convert merges in model to map")
		return errors.New("could not convert merges in model to map")
	}

	// Convert the slice of interfaces to a slice of strings
	var mergesStr []string
	for _, v := range merges {
		mergesStr = append(mergesStr, v.(string))
	}

	mergesPath := path.Join(*dir, "merges.txt")

	// Create the file
	mergesFile, err := os.Create(mergesPath)
	if err != nil {
		log.Println("Error creating file:", err)
		return err
	}
	defer func(mergesFile *os.File) {
		_ = mergesFile.Close()
	}(mergesFile)

	// Write each merge string to a new line in the file
	for _, v := range merges {
		_, err = mergesFile.WriteString(v.(string) + "\n")
		if err != nil {
			log.Println("Error writing to file:", err)
			return err
		}
	}

	log.Println("Merges written to merges.txt from tokenizer.json")

	if mmapErr := resources.AddEntry(
		"merges.txt", mergesFile,
	); mmapErr != nil {
		return fmt.Errorf("error trying to mmap file: %s", mmapErr)
	}

	return nil
}

func FindNumberOfShardsFromConfig(configPath string) (int, error) {
	// Open the file
	configFile, err := os.Open(configPath)
	if err != nil {
		log.Println("Error opening config:", err)
		return -1, err
	}
	defer func(configFile *os.File) {
		_ = configFile.Close()
	}(configFile)

	// Decode the JSON data into a map
	var data JsonMap
	err = json.NewDecoder(configFile).Decode(&data)
	if err != nil {
		log.Println("Error decoding JSON from config:", err)
		return -1, err
	}

	// Access the data at the specified path
	weightMap, ok := data["weight_map"].(map[string]interface{})
	weightMap = ToJsonMap(weightMap)
	if !ok {
		fmt.Println("Error: Could not convert data to weight_map")
		return -1, errors.New("could not convert data to weight_map")
	}
	// Try embed out, if not, try lm_head.weight
	nameOfLast, ok := weightMap["embed_out.weight"]
	if !ok {
		nameOfLast, ok = weightMap["lm_head.weight"]
		if !ok {
			fmt.Println("Error: Could not convert weight_map to embed_out or lm_head")
			return -1, errors.New("could not convert weight_map to embed_out or lm_head")
		}
	}

	r, _ := regexp.Compile(`\D*\d+\D+(\d+)`)
	// convert to interface -> string -> int
	nameOfLastInt, err := strconv.Atoi(
		r.FindStringSubmatch(fmt.Sprintf("%v", nameOfLast))[1],
	)

	if err != nil {
		fmt.Println("Error: Could not convert embed_out to int")
		return -1, errors.New("could not convert embed_out to int")
	}

	return nameOfLastInt, nil
}

func FindProcessingStepsFromTokenizer(model ResourceEntry) (
	[]Processor,
	error,
) {
	// convert the data to a map
	var data JsonMap
	err := json.Unmarshal(*model.Data, &data)
	if err != nil {
		return nil, err
	}

	// create array of processors
	var processors []Processor
	// check if normalizer is present
	normalizer, ok := data["normalizer"].(map[string]interface{})
	normalizer = ToJsonMap(normalizer)
	if normalizer != nil && ok {
		// add normalizer to processors
		processor := Processor{
			ProcessorType: "normalizer",
			ProcessorArgs: normalizer,
		}
		processors = append(processors, processor)
	}
	// check if pre_tokenizer is present
	preTokenizer, ok := data["pre_tokenizer"].(map[string]interface{})
	preTokenizer = ToJsonMap(preTokenizer)
	if preTokenizer != nil && ok {
		// add pre_tokenizer to processors
		processor := Processor{
			ProcessorType: "pre_tokenizer",
			ProcessorArgs: preTokenizer,
		}
		processors = append(processors, processor)
	}
	// check if post_processor is present
	post_processor, ok := data["post_processor"].(map[string]interface{})
	post_processor = ToJsonMap(post_processor)
	if post_processor != nil && ok {
		// add post_processor to processors
		processor := Processor{
			ProcessorType: "post_processor",
			ProcessorArgs: post_processor,
		}
		processors = append(processors, processor)
	}
	// check if decoder is present
	decoder, ok := data["decoder"].(map[string]interface{})
	decoder = ToJsonMap(decoder)
	if decoder != nil && ok {
		// add decoder to processors
		processor := Processor{
			ProcessorType: "decoder",
			ProcessorArgs: decoder,
		}
		processors = append(processors, processor)
	}

	return processors, nil
}

func ToJsonMap(sim map[string]interface{}) JsonMap {
	jm := make(JsonMap)
	for k, v := range sim {
		jm[k] = v
	}
	return jm
}

func (jm JsonMap) ToMapInterface() map[string]interface{} {
	m := make(map[string]interface{})
	for k, v := range jm {
		m[k] = v
	}
	return m
}

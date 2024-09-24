gpt-bpe
=======
An implementation of GPT2 Byte Pair Encoding Encoder/Decoder in `golang`.  Generally very fast, bottlenecked by the regex engine for whitespace separation.

## Building for Gopherjs

gpt_bpe can be transpiled to javascript to be embedded into web pages, like in goose-web. To build for this purpose, run `make gopherBuild` and copy the generated files to your javascript application. 

Furthur implementation details can be found in goose-web's tokenizer implementation.
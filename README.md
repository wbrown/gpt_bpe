gpt-bpe
=======
An implementation of GPT2 Byte Pair Encoding Encoder/Decoder in `golang`.  Generally very fast, bottlenecked by the regex engine for whitespace separation.

To compile for wasm/wasip1 target. After it's compiled, it will be usable with any language that has extism host SDK support(https://extism.org/docs/concepts/host-sdk).

```
cd wasm
./build.sh
```


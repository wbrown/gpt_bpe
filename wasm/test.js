import fs from 'fs';
import createPlugin from '@extism/extism';
import msgpack from "msgpack-lite";

const corpus = fs.readFileSync("../resources/frankenstein.txt", 'utf8');

async function main() {
  try {
    const plugin = await createPlugin(
      'tok.wasm',
      { useWasi: true }
    );

    console.log("Tokenizing sample text...");
    const sampleText = "Hello, world! This is a test.";
    const sampleResult = await plugin.call("tokenize", sampleText);
    
    const decodedSample = msgpack.decode(sampleResult.bytes());
    console.log("Sample tokens:", decodedSample);
    
    if (decodedSample.tokens && decodedSample.tokens.length > 0) {
      const tokenBytes = msgpack.encode(decodedSample);
      const decodedText = await plugin.call("decode", tokenBytes);
      console.log("Decoded text:", decodedText.text());
    }

    console.log("\nRunning benchmark...");
    const times = [];
    for (let i = 0; i < 100; i++) {
      const start = process.hrtime.bigint();
      
      const result = await plugin.call("tokenize", corpus);
      
      const end = process.hrtime.bigint();
      const duration = Number(end - start) / 1000000;
      times.push(duration);
      
      if (i === 0) {
        const decoded = msgpack.decode(result.bytes());
        console.log(`Corpus tokenized into ${decoded.tokens.length} tokens`);
      }
    }
    
    // stats
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    
    console.log(`\nBenchmark results (100 iterations):`);
    console.log(`Average: ${avgTime.toFixed(2)}ms`);
    console.log(`Min: ${minTime.toFixed(2)}ms`);
    console.log(`Max: ${maxTime.toFixed(2)}ms`);

  } catch(err) {
    console.error("Error:", err);
  }
}

main()
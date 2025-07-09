import fs from 'fs';
import { createTokenizer } from './bindings.js';

const corpus = fs.readFileSync("../resources/frankenstein.txt", 'utf8');

async function main() {
  try {
    const tokenizer = await createTokenizer();
    console.log("Tokenizer created");

    const sampleText = "Hello, world! This is a test.";
    
    const tokens = await tokenizer.tokenize(sampleText);
    console.log("Sample tokens:", tokens);
    
    if (tokens && tokens.length > 0) {
      const decodedText = await tokenizer.decode(tokens);
      console.log("Decoded text:", decodedText);
    }

    console.log("\nRunning benchmark...");
    const times = [];
    for (let i = 0; i < 100; i++) {
      const start = process.hrtime.bigint();
      
      const result = await tokenizer.tokenize(corpus);
      
      const end = process.hrtime.bigint();
      const duration = Number(end - start) / 1000000;
      times.push(duration);
      
      if (i === 0) {
        console.log(`Corpus tokenized into ${result.length} tokens`);
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

main(); 
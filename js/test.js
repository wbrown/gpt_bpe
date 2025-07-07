const fs = require('fs');

const corpus = fs.readFileSync("../resources/frankenstein.txt");

function sleep(ms) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

const getMethods = (obj) => {
    let properties = new Set()
    let currentObj = obj
    do {
        Object.getOwnPropertyNames(currentObj).map(item => properties.add(item))
    } while ((currentObj = Object.getPrototypeOf(currentObj)))
    return [...properties.keys()]
}

async function main() {
  try {
    const gpt_bpe_obj = await import('./gpt_bpe.js');
    await sleep(1000);
    
    console.log("Tokenizing sample text...");
    const sampleText = "Hello, world! This is a test.";
    const sampleTokens = gpt_bpe_obj.default.tokenize(sampleText);
    console.log("Sample tokens:", sampleTokens);
    
    if (sampleTokens && sampleTokens.length > 0) {
      const decodedText = gpt_bpe_obj.default.decode(sampleTokens);
      console.log("Decoded text:", decodedText);
    }

    console.log("\nRunning benchmark...");
    const times = [];
    for (let i = 0; i < 100; i++) {
      const start = process.hrtime.bigint();
      
      tokens = gpt_bpe_obj.default.tokenize(corpus)
      
      const end = process.hrtime.bigint();
      const duration = Number(end - start) / 1000000;
      times.push(duration);
      
      if (i === 0) {
        console.log(`Corpus tokenized into ${tokens.length} tokens`);
      }
    }
    
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    
    console.log(`\nBenchmark results (100 iterations):`);
    console.log(`Average: ${avgTime.toFixed(2)}ms`);
    console.log(`Min: ${minTime.toFixed(2)}ms`);
    console.log(`Max: ${maxTime.toFixed(2)}ms`);
  }
  catch(err) {
    console.log(err);
  }
}

main()

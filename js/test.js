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
    for (let i = 0; i < 100; i++) {
          console.time("gpt_bpe");
          tokens = gpt_bpe_obj.default.tokenize(corpus)
          console.timeEnd("gpt_bpe");
      }
  }
  catch(err) {
    console.log(err);
  }
}

main()

import msgpack from "msgpack-lite";

/**
 * JavaScript bindings for the gpt-bpe wasm module
 */
export class GPTBPETokenizer {
  constructor(plugin) {
    this.plugin = plugin;
  }

  /**
   * Tokenize text into an array of token IDs
   * @param {string} text - The text to tokenize
   * @returns {Promise<number[]>} Array of token IDs
   */
  async tokenize(text) {
    const result = await this.plugin.call("tokenize", text);
    return msgpack.decode(result.bytes());
  }

  /**
   * Tokenizes text and decodes it back for round-trip testing
   * @param {string} text - Input text
   * @returns {Promise<string>} Round-tripped text
   */
  async tokenizeAndBack(text) {
    const result = await this.plugin.call("tokenize_and_back", text);
    return result.text();
  }

  /**
   * Decodes a number[] array of token IDs to text.
   * @param {number[]} tokens - Array of token IDs
   * @returns {Promise<string>} The decoded text
   */
  async decodeArray(tokens) {
    const tokenBytes = msgpack.encode(tokens);
    const result = await this.plugin.call("decode_array", tokenBytes);
    return result.text();
  }

  /**
   * Decode packed token data back to text
   * @param {Uint8Array} packedTokens - Packed token bytes
   * @returns {Promise<string>} The decoded text
   */
  async decode(packedTokens) {
    const result = await this.plugin.call("decode", packedTokens);
    return result.text();
  }

  /**
   * Get the length of tokenized text without returning the tokens
   * @param {string} text - The text to tokenize
   * @returns {Promise<number>} Number of tokens
   */
  async getTokenCount(text) {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }
}

/**
 * Factory function to create a tokenizer instance
 * @param {string} wasmPath - Path to the WASM file
 * @param {Object} options - Options for the Extism plugin
 * @returns {Promise<GPTBPETokenizer>} Tokenizer instance
 */
export async function createTokenizer(wasmPath = 'tok.wasm', options = { useWasi: true }) {
  const createPlugin = (await import('@extism/extism')).default;
  const plugin = await createPlugin(wasmPath, options);
  return new GPTBPETokenizer(plugin);
}

export default GPTBPETokenizer;
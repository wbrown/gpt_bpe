import brotli
import os

def compress_wasm_file():
    with open('tok.wasm', 'rb') as f:
        wasm_data = f.read()
    
    print(f"Original file size: {len(wasm_data)} bytes")
    
    compressed_data = brotli.compress(wasm_data)
    
    print(f"Compressed file size: {len(compressed_data)} bytes")
    print(f"Compression ratio: {len(compressed_data) / len(wasm_data):.2%}")
    
    with open('tok_slim.wasm', 'wb') as f:
        f.write(compressed_data)
    
    print("Compression complete! Output saved as tok_slim.wasm")

def decompress_wasm_file():
    with open('tok_slim.wasm', 'rb') as f:
        compressed_data = f.read()
    
    decompressed_data = brotli.decompress(compressed_data)
    
    with open('tok_slim_decompressed.wasm', 'wb') as f:
        f.write(decompressed_data)

if __name__ == "__main__":
    compress_wasm_file()
    # decompress_wasm_file()


import ctypes
import numpy
import typing
from typing import Union, Sequence, Type

gpt_bpe = ctypes.cdll.LoadLibrary("./gpt_bpe.dylib")


class Tokens(ctypes.Structure):
    _fields_ = [("tokens", ctypes.c_void_p), ("len", ctypes.c_uint64)]

    def __del__(self):
        gpt_bpe.freeTokens(self)


class BackedArray(numpy.ndarray):
    def __new__(
        subtype,
        shape,
        dtype: Type = float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        backed=None,
    ):
        obj = super().__new__(subtype, shape, dtype, buffer, offset, strides, order)
        # set the new 'info' attribute to the value passed
        obj.backed = backed
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.backed = getattr(obj, "backed", None)


class BPETokenizer:
    def __init__(self, vocab_id: str):
        self.vocab_id = vocab_id.encode("utf8")
        gpt_bpe.initTokenizer(self.vocab_id)
        gpt_bpe.tokenize.restype = Tokens
        gpt_bpe.decode.restype = ctypes.c_char_p

    def encode(self, text: str) -> numpy.ndarray:
        encoded = text.encode("utf8")
        tokens_struct = gpt_bpe.tokenize(self.vocab_id, encoded)
        tokens_arr_type = ctypes.c_uint32 * tokens_struct.len
        tokens_buf = tokens_arr_type.from_address(tokens_struct.tokens)
        return BackedArray(
            [len(tokens_buf)],
            dtype=ctypes.c_uint32,
            buffer=tokens_buf,
            backed=tokens_struct,
        )

    def decode(self, arr: Union[numpy.ndarray, Sequence[int]]) -> str:
        if type(arr) == numpy.ndarray and arr.dtype != ctypes.c_uint32:
            arr = arr.astype(ctypes.c_uint32)
        elif type(arr) == BackedArray:
            pass
        elif type(arr) != numpy.ndarray:
            arr = numpy.array(arr, dtype=ctypes.c_uint32)
        tokens = Tokens()
        tokens.len = len(arr)
        tokens.tokens = ctypes.c_void_p(arr.ctypes.data)
        return gpt_bpe.decode(self.vocab_id, tokens)


encoder = BPETokenizer("gpt2-tokenizer")

test_str = "This is a test."
tokens = encoder.encode(test_str)

print(tokens)

print(encoder.decode(tokens))
print(encoder.decode([1212, 318, 257, 1332, 13]))

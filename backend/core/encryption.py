"""
Heimdall - Paillier Homomorphic Encryption Module
==================================================
Handles key generation, encryption, and decryption.

Cross-Library Interoperability
-------------------------------
Python's `phe` library uses BASE=16 for its fixed-point encoding scheme:
    decoded_value = raw_integer × 16^exponent

JavaScript Paillier libraries (and this project's crypto.js) use BASE=10:
    decoded_value = raw_integer × 10^exponent

These conventions are incompatible if mixed naively. phe's own documentation
(phe/encoding.py) explicitly describes the fix:
    "If working with other Paillier libraries you will have to agree on a
     specific BASE and LOG2_BASE — inheriting from this class and overriding
     those two attributes will enable this."

We do exactly that: `Base10EncodedNumber` subclasses `EncodedNumber` with
BASE=10, making phe's arithmetic produce and consume exponents in decimal,
matching the JS wire format perfectly.

Wire Format (Heimdall)
----------------------
{
  "ciphertext": str,   # decimal integer in Z_{n²}
  "exponent":   int,   # signed; decoded_value = integer × 10^exponent
}
"""

import math
import phe as paillier
from phe.encoding import EncodedNumber
from typing import Tuple


# ---------------------------------------------------------------------------
# BASE=10 Encoder — the core interop fix
# ---------------------------------------------------------------------------

class Base10EncodedNumber(EncodedNumber):
    """
    Paillier fixed-point encoder using BASE=10 (decimal).

    phe's default EncodedNumber uses BASE=16. JavaScript Paillier libraries
    scale by powers of 10 (e.g. value × 10^6, exponent = -6). Using BASE=16
    on the Python side and BASE=10 on the JS side causes a silent magnitude
    error — empirically shown to produce results billions of times wrong.

    By subclassing EncodedNumber and setting BASE=10, ALL phe arithmetic
    (encode, decode, decrease_exponent_to) uses decimal scaling, matching
    the JS convention exactly. No monkey-patching required.
    """
    BASE = 10
    LOG2_BASE = math.log(10, 2)


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def generate_keys(key_size: int = 2048) -> Tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    """Generate a Paillier key pair."""
    public_key, private_key = paillier.generate_paillier_keypair(n_length=key_size)
    return public_key, private_key


def serialize_public_key(public_key: paillier.PaillierPublicKey) -> dict:
    return {"n": str(public_key.n)}


def deserialize_public_key(data: dict) -> paillier.PaillierPublicKey:
    return paillier.PaillierPublicKey(n=int(data["n"]))


# ---------------------------------------------------------------------------
# Encryption / Decryption (BASE=10, Python-side)
# ---------------------------------------------------------------------------

def encrypt_value(public_key: paillier.PaillierPublicKey, value: float) -> dict:
    """Encrypt a single float using BASE=10 encoding, compatible with JS wire format."""
    encoded = Base10EncodedNumber.encode(public_key, value)
    enc = public_key.encrypt(encoded)
    return {"ciphertext": str(enc.ciphertext()), "exponent": enc.exponent}


def encrypt_vector(public_key: paillier.PaillierPublicKey, values: list) -> list:
    """Encrypt a list of floats using BASE=10 encoding."""
    return [encrypt_value(public_key, v) for v in values]


def decrypt_value(private_key: paillier.PaillierPrivateKey, enc_dict: dict) -> float:
    """Decrypt a single encrypted value using BASE=10 decoding."""
    public_key = private_key.public_key
    enc_number = paillier.EncryptedNumber(
        public_key,
        int(enc_dict["ciphertext"]),
        int(enc_dict["exponent"]),
    )
    return private_key.decrypt_encoded(enc_number, Encoding=Base10EncodedNumber).decode()


# ---------------------------------------------------------------------------
# Reconstruct an EncryptedNumber from a JS-generated ciphertext
# ---------------------------------------------------------------------------

def reconstruct_encrypted_number(public_key: paillier.PaillierPublicKey, enc_dict: dict):
    """
    Wrap a JS-generated ciphertext as a phe EncryptedNumber.

    The caller must pass Base10EncodedNumber scalars to all arithmetic
    operations so phe's exponent tracking stays in BASE=10.
    """
    return paillier.EncryptedNumber(
        public_key,
        int(enc_dict["ciphertext"]),
        int(enc_dict["exponent"]),
    )

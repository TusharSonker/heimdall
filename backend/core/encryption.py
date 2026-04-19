"""
Heimdall - Paillier Homomorphic Encryption Module
Handles key generation, encryption, and decryption.
"""

import phe as paillier
import json
import base64
from typing import Tuple


def generate_keys(key_size: int = 2048) -> Tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    """Generate a Paillier key pair."""
    public_key, private_key = paillier.generate_paillier_keypair(n_length=key_size)
    return public_key, private_key


def serialize_public_key(public_key: paillier.PaillierPublicKey) -> dict:
    """Serialize a public key to a JSON-compatible dict."""
    return {"n": str(public_key.n)}


def deserialize_public_key(data: dict) -> paillier.PaillierPublicKey:
    """Deserialize a public key from a dict."""
    return paillier.PaillierPublicKey(n=int(data["n"]))


def encrypt_vector(public_key: paillier.PaillierPublicKey, values: list[float]) -> list[dict]:
    """
    Encrypt a list of floats using the Paillier public key.
    Returns a list of serialized ciphertexts.
    """
    encrypted = []
    for v in values:
        enc = public_key.encrypt(v)
        encrypted.append({
            "ciphertext": str(enc.ciphertext()),
            "exponent": enc.exponent
        })
    return encrypted


def decrypt_value(private_key: paillier.PaillierPrivateKey, enc_dict: dict) -> float:
    """
    Decrypt a single encrypted value.
    enc_dict: {"ciphertext": str, "exponent": int}
    """
    public_key = private_key.public_key
    enc_number = paillier.EncryptedNumber(
        public_key,
        int(enc_dict["ciphertext"]),
        int(enc_dict["exponent"])
    )
    return private_key.decrypt(enc_number)


def reconstruct_encrypted_number(public_key, enc_dict: dict):
    """
    Reconstruct EncryptedNumber from JS-generated ciphertext.
    
    Python phe stores ciphertexts differently from raw Paillier.
    We must set decrease_exponent_to=None to prevent phe from
    re-encoding the ciphertext internally.
    """
    enc = paillier.EncryptedNumber(
        public_key,
        int(enc_dict["ciphertext"]),
        int(enc_dict["exponent"])
    )
    # Prevent phe from touching the exponent during arithmetic
    enc.decrease_exponent_to = lambda x: None
    return enc
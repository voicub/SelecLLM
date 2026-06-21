"""core/encryption.py — real ChaCha20-Poly1305 E2EE with Curve25519 ECDH."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import os as _os, struct, time
from dataclasses import dataclass
from typing import Tuple
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives import serialization
from config import CFG
from utils.logger import get_logger
log = get_logger("E2EE")

# Packet wire format: [4B magic][4B seq][12B nonce][4B plen][ciphertext+16B tag]
_MAGIC   = b"SLEC"
_HDR_FMT = "!4sI12sI"
_HDR_SZ  = struct.calcsize(_HDR_FMT)   # 24 bytes

@dataclass
class EncryptedPacket:
    seq_no: int; nonce: bytes; ciphertext: bytes; raw_bytes: bytes
    @property
    def byte_size(self): return len(self.raw_bytes)

@dataclass
class DecryptedPacket:
    seq_no: int; plaintext: bytes; latency_ms: float

class E2EETunnel:
    """Bidirectional ChaCha20-Poly1305 tunnel with Curve25519 key exchange."""
    def __init__(self, name="node"):
        self.name=name; self._priv=X25519PrivateKey.generate()
        self._seq=0; self._cc=None
    @property
    def public_key_bytes(self):
        return self._priv.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    def set_peer_public_key(self, peer_pub: bytes):
        shared=self._priv.exchange(X25519PublicKey.from_public_bytes(peer_pub))
        self._cc=ChaCha20Poly1305(shared[:32])
        log.debug(f"'{self.name}': ECDH done, key derived")
    def encrypt(self, plaintext: bytes, aad: bytes = b"") -> EncryptedPacket:
        if self._cc is None: raise RuntimeError("Key exchange not complete")
        t0=time.perf_counter()
        nonce=_os.urandom(CFG.chacha_nonce_bytes)
        ct=self._cc.encrypt(nonce,plaintext,aad or None)
        seq=self._seq; self._seq+=1
        hdr=struct.pack(_HDR_FMT,_MAGIC,seq,nonce,len(ct))
        raw=hdr+ct
        log.debug(f"enc seq={seq} {len(plaintext)}B→{len(raw)}B {(time.perf_counter()-t0)*1e6:.0f}µs")
        return EncryptedPacket(seq,nonce,ct,raw)
    def decrypt(self, raw: bytes, aad: bytes = b"") -> DecryptedPacket:
        if self._cc is None: raise RuntimeError("Key exchange not complete")
        t0=time.perf_counter()
        magic,seq,nonce,plen=struct.unpack(_HDR_FMT,raw[:_HDR_SZ])
        if magic!=_MAGIC: raise ValueError(f"Bad magic: {magic!r}")
        pt=self._cc.decrypt(nonce,raw[_HDR_SZ:_HDR_SZ+plen],aad or None)
        return DecryptedPacket(seq,pt,(time.perf_counter()-t0)*1000)
    @staticmethod
    def establish_pair(na="sender",nb="receiver") -> Tuple["E2EETunnel","E2EETunnel"]:
        a=E2EETunnel(na); b=E2EETunnel(nb)
        a.set_peer_public_key(b.public_key_bytes)
        b.set_peer_public_key(a.public_key_bytes)
        return a,b

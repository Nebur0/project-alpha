"""Notifiche Web Push (standard RFC 8030/8291/8292) senza servizi di terzi.

- chiavi VAPID generate in locale e salvate fuori dal repository versionato
  (``data/push/``, ignorata da git);
- messaggi cifrati end-to-end (aes128gcm): il servizio push di Apple o
  Google li consegna senza poterli leggere;
- iscrizioni dei dispositivi in un file JSON, accettate solo verso i servizi
  push noti dei browser.

La cifratura è verificata contro l'esempio dell'RFC 8291 (Appendice A).
"""

from __future__ import annotations

import base64
import json
import os
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from project_alpha.io_utils import write_text_atomic

# Servizi push dei browser: le iscrizioni verso altri indirizzi sono rifiutate.
PUSH_SERVICE_SUFFIXES = (
    "push.apple.com",
    "fcm.googleapis.com",
    "push.services.mozilla.com",
    "notify.windows.com",
)
RECORD_SIZE = 4096
MAX_PAYLOAD = 3000  # margine sotto i ~4 KB consentiti dai servizi push
DEFAULT_CONTACT = "https://github.com/Nebur0/project-alpha"


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _hkdf(salt: bytes, ikm: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def _public_bytes(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )


def encrypt(
    plaintext: bytes,
    user_agent_public: bytes,
    auth_secret: bytes,
    *,
    sender_key: ec.EllipticCurvePrivateKey | None = None,
    salt: bytes | None = None,
) -> bytes:
    """Cifra un messaggio per un dispositivo (RFC 8291, codifica aes128gcm)."""
    if len(plaintext) > MAX_PAYLOAD:
        raise ValueError("Messaggio troppo lungo per una notifica push.")

    sender = sender_key or ec.generate_private_key(ec.SECP256R1())
    salt = salt or os.urandom(16)
    sender_public = _public_bytes(sender)
    receiver = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), user_agent_public)
    shared = sender.exchange(ec.ECDH(), receiver)

    ikm = _hkdf(auth_secret, shared, b"WebPush: info\x00" + user_agent_public + sender_public, 32)
    content_key = _hkdf(salt, ikm, b"Content-Encoding: aes128gcm\x00", 16)
    nonce = _hkdf(salt, ikm, b"Content-Encoding: nonce\x00", 12)
    ciphertext = AESGCM(content_key).encrypt(nonce, plaintext + b"\x02", None)

    header = salt + struct.pack("!IB", RECORD_SIZE, len(sender_public)) + sender_public
    return header + ciphertext


@dataclass(frozen=True)
class VapidKeys:
    private_key: ec.EllipticCurvePrivateKey

    @property
    def public_key_b64(self) -> str:
        """Chiave pubblica da dare al browser (``applicationServerKey``)."""
        return b64url_encode(_public_bytes(self.private_key))

    @classmethod
    def load_or_create(cls, directory: Path | str) -> VapidKeys:
        path = Path(directory) / "vapid_private.pem"
        if path.exists():
            key = serialization.load_pem_private_key(path.read_bytes(), password=None)
            if not isinstance(key, ec.EllipticCurvePrivateKey):
                raise ValueError("Chiave VAPID non valida.")
            return cls(key)

        key = ec.generate_private_key(ec.SECP256R1())
        pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        write_text_atomic(path, pem.decode("ascii"))
        return cls(key)

    def authorization(self, endpoint: str, *, contact: str, now: float | None = None) -> str:
        """Intestazione ``Authorization`` VAPID (RFC 8292, firma ES256)."""
        parsed = urlparse(endpoint)
        claims = {
            "aud": f"{parsed.scheme}://{parsed.netloc}",
            "exp": int((now or time.time()) + 12 * 3600),
            "sub": contact,
        }
        header = {"typ": "JWT", "alg": "ES256"}
        signing_input = (
            b64url_encode(json.dumps(header, separators=(",", ":")).encode())
            + "."
            + b64url_encode(json.dumps(claims, separators=(",", ":")).encode())
        )
        der = self.private_key.sign(signing_input.encode("ascii"), ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        token = f"{signing_input}.{b64url_encode(signature)}"
        return f"vapid t={token}, k={self.public_key_b64}"


# ----------------------------------------------------------------------
# Iscrizioni
# ----------------------------------------------------------------------


def validate_subscription(payload: Any) -> dict[str, Any]:
    """Controlla un'iscrizione ricevuta dal browser; solleva ValueError se non valida."""
    if not isinstance(payload, Mapping):
        raise ValueError("Iscrizione non valida.")

    endpoint = payload.get("endpoint")
    keys = payload.get("keys")
    if not isinstance(endpoint, str) or not isinstance(keys, Mapping):
        raise ValueError("Iscrizione non valida.")

    parsed = urlparse(endpoint)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not any(
        host == suffix or host.endswith("." + suffix) for suffix in PUSH_SERVICE_SUFFIXES
    ):
        raise ValueError("Servizio push non riconosciuto.")

    try:
        p256dh = b64url_decode(str(keys.get("p256dh", "")))
        auth = b64url_decode(str(keys.get("auth", "")))
        ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), p256dh)
    except (ValueError, TypeError) as exc:
        raise ValueError("Chiavi dell'iscrizione non valide.") from exc

    if len(auth) != 16:
        raise ValueError("Chiavi dell'iscrizione non valide.")

    return {"endpoint": endpoint, "keys": {"p256dh": keys["p256dh"], "auth": keys["auth"]}}


class SubscriptionStore:
    """Iscrizioni dei dispositivi in un file JSON (una per endpoint)."""

    def __init__(self, directory: Path | str) -> None:
        self.path = Path(directory) / "subscriptions.json"

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return list(json.loads(self.path.read_text(encoding="utf-8")))

    def _save(self, items: list[dict[str, Any]]) -> None:
        write_text_atomic(self.path, json.dumps(items, indent=2))

    def add(self, subscription: Mapping[str, Any], *, label: str = "") -> None:
        clean = validate_subscription(subscription)
        items = [item for item in self.all() if item["endpoint"] != clean["endpoint"]]
        items.append({**clean, "label": label[:60], "added": int(time.time())})
        self._save(items)

    def remove(self, endpoint: str) -> bool:
        items = self.all()
        kept = [item for item in items if item["endpoint"] != endpoint]
        if len(kept) != len(items):
            self._save(kept)
            return True
        return False


# ----------------------------------------------------------------------
# Invio
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DeliveryResult:
    sent: int
    removed: int
    failed: tuple[str, ...]


def send_notification(
    store: SubscriptionStore,
    keys: VapidKeys,
    message: Mapping[str, Any],
    *,
    contact: str = DEFAULT_CONTACT,
    urgency: str = "normal",
    ttl: int = 24 * 3600,
    post: Callable[..., Any] | None = None,
) -> DeliveryResult:
    """Invia ``message`` (JSON) a tutti i dispositivi iscritti.

    Le iscrizioni scadute (HTTP 404/410) vengono rimosse.
    """
    sender = post or requests.post
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    sent = removed = 0
    failed: list[str] = []

    for item in store.all():
        endpoint = item["endpoint"]
        payload = encrypt(
            body, b64url_decode(item["keys"]["p256dh"]), b64url_decode(item["keys"]["auth"])
        )
        headers = {
            "Authorization": keys.authorization(endpoint, contact=contact),
            "Content-Encoding": "aes128gcm",
            "Content-Type": "application/octet-stream",
            "TTL": str(ttl),
            "Urgency": urgency,
        }
        try:
            response = sender(endpoint, data=payload, headers=headers, timeout=20)
        except requests.RequestException as exc:
            failed.append(f"{urlparse(endpoint).hostname}: {type(exc).__name__}")
            continue

        if response.status_code in (404, 410):
            store.remove(endpoint)
            removed += 1
        elif response.status_code >= 400:
            failed.append(f"{urlparse(endpoint).hostname}: HTTP {response.status_code}")
        else:
            sent += 1

    return DeliveryResult(sent=sent, removed=removed, failed=tuple(failed))

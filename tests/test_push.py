"""Test delle notifiche Web Push e degli endpoint della web app (nessuna rete esterna)."""

import json
import shutil
import struct
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from project_alpha.dashboard.server import create_server, origin_allowed
from project_alpha.monitor import Alert, MonitorResult
from project_alpha.notifications import build_message
from project_alpha.push import (
    SubscriptionStore,
    VapidKeys,
    b64url_decode,
    b64url_encode,
    encrypt,
    send_notification,
    validate_subscription,
)

REPO = Path(__file__).resolve().parents[1]
APPLE = "https://web.push.apple.com/QGuQyavXutnMH-test"


def _device():
    key = ec.generate_private_key(ec.SECP256R1())
    public = key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    auth = b"0123456789abcdef"
    subscription = {
        "endpoint": APPLE,
        "keys": {"p256dh": b64url_encode(public), "auth": b64url_encode(auth)},
    }
    return key, public, auth, subscription


def _decrypt(body: bytes, key, public: bytes, auth: bytes) -> bytes:
    """Lato dispositivo (RFC 8291): serve a verificare la cifratura."""
    salt, _rs, idlen = body[:16], *struct.unpack("!IB", body[16:21])
    sender_public = body[21 : 21 + idlen]
    ciphertext = body[21 + idlen :]
    sender = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), sender_public)
    shared = key.exchange(ec.ECDH(), sender)

    def hkdf(salt_, ikm, info, n):
        return HKDF(hashes.SHA256(), n, salt_, info).derive(ikm)

    ikm = hkdf(auth, shared, b"WebPush: info\x00" + public + sender_public, 32)
    cek = hkdf(salt, ikm, b"Content-Encoding: aes128gcm\x00", 16)
    nonce = hkdf(salt, ikm, b"Content-Encoding: nonce\x00", 12)
    plain = AESGCM(cek).decrypt(nonce, ciphertext, None)
    assert plain.endswith(b"\x02")
    return plain[:-1]


# ----------------------------------------------------------------------
# Cifratura e VAPID
# ----------------------------------------------------------------------


def test_encryption_matches_rfc8291_example() -> None:
    sender = ec.derive_private_key(
        int.from_bytes(b64url_decode("yfWPiYE-n46HLnH0KqZOF1fJJU3MYrct3AELtAQ-oRw"), "big"),
        ec.SECP256R1(),
    )

    body = encrypt(
        b"When I grow up, I want to be a watermelon",
        b64url_decode(
            "BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4"
        ),
        b64url_decode("BTBZMqHH6r4Tts7J_aSIgg"),
        sender_key=sender,
        salt=b64url_decode("DGv6ra1nlYgDCS1FRnbzlw"),
    )

    assert b64url_encode(body) == (
        "DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLocInmYWA"
        "mS6TlzAC8wEqKK6PBru3jl7A_yl95bQpu6cVPTpK4Mqgkf1CXztLVBSt2Ks3oZwbuwXPXLWyouBWLVWGNWQexSgSx"
        "sj_Qulcy4a-fN"
    )


def test_encrypted_message_round_trip() -> None:
    key, public, auth, _ = _device()

    body = encrypt("Capitale 10.000 $ ✅".encode(), public, auth)

    assert _decrypt(body, key, public, auth).decode() == "Capitale 10.000 $ ✅"


def test_vapid_keys_are_persisted_and_jwt_is_valid(tmp_path) -> None:
    keys = VapidKeys.load_or_create(tmp_path)
    again = VapidKeys.load_or_create(tmp_path)
    assert keys.public_key_b64 == again.public_key_b64

    header = keys.authorization(APPLE, contact="https://example.org", now=1_000_000)
    token = header.split("t=", 1)[1].split(",", 1)[0]
    head, claims, signature = token.split(".")
    payload = json.loads(b64url_decode(claims))
    assert payload == {
        "aud": "https://web.push.apple.com",
        "exp": 1_043_200,
        "sub": "https://example.org",
    }
    assert header.endswith("k=" + keys.public_key_b64)

    raw = b64url_decode(signature)
    der = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
    keys.private_key.public_key().verify(
        der, f"{head}.{claims}".encode(), ec.ECDSA(hashes.SHA256())
    )
    with pytest.raises(InvalidSignature):
        keys.private_key.public_key().verify(der, b"altro", ec.ECDSA(hashes.SHA256()))


# ----------------------------------------------------------------------
# Iscrizioni e invio
# ----------------------------------------------------------------------


def test_subscription_validation() -> None:
    _, _, _, subscription = _device()
    assert validate_subscription(subscription)["endpoint"] == APPLE

    for endpoint in ("https://evil.example.com/x", "http://web.push.apple.com/x", "file:///etc"):
        with pytest.raises(ValueError):
            validate_subscription({**subscription, "endpoint": endpoint})

    with pytest.raises(ValueError):
        validate_subscription({**subscription, "keys": {"p256dh": "AAAA", "auth": "AAAA"}})


def test_store_deduplicates_and_removes(tmp_path) -> None:
    store = SubscriptionStore(tmp_path)
    _, _, _, subscription = _device()

    store.add(subscription, label="iPhone")
    store.add(subscription, label="iPhone")
    assert len(store.all()) == 1
    assert store.remove(APPLE) and store.all() == []


class FakeResponse:
    def __init__(self, status):
        self.status_code = status


def test_send_encrypts_signs_and_drops_expired(tmp_path) -> None:
    store = SubscriptionStore(tmp_path)
    key, public, auth, subscription = _device()
    store.add(subscription)
    expired = {**_device()[3], "endpoint": "https://fcm.googleapis.com/fcm/send/old"}
    store.add(expired)
    calls = []

    def post(url, data, headers, timeout):
        calls.append((url, data, headers))
        return FakeResponse(201 if url == APPLE else 410)

    result = send_notification(
        store, VapidKeys.load_or_create(tmp_path), {"title": "T", "body": "B"}, post=post
    )

    assert (result.sent, result.removed) == (1, 1)
    assert [item["endpoint"] for item in store.all()] == [APPLE]
    url, data, headers = next(call for call in calls if call[0] == APPLE)
    assert headers["Content-Encoding"] == "aes128gcm"
    assert headers["Authorization"].startswith("vapid t=")
    assert json.loads(_decrypt(data, key, public, auth)) == {"title": "T", "body": "B"}


# ----------------------------------------------------------------------
# Messaggio
# ----------------------------------------------------------------------


def test_message_for_regular_and_critical_runs() -> None:
    last = {
        "status": "SUBMITTED",
        "order_count": 10,
        "equity": 10012.4,
        "daily_return": 0.0012,
        "drawdown": 0.0,
    }
    ok = build_message(MonitorResult("t", (), last, None))
    assert ok["title"] == "✅ Project Alpha"
    assert ok["body"].startswith(
        "Ordini inviati, eseguiti all'apertura (10). Capitale 10.012 $ (+0,12%)"
    )
    assert ok["urgency"] == "normal"

    stop = build_message(
        MonitorResult(
            "t",
            (
                Alert("critical", "HALTED", "Sistema fermo dopo lo stop."),
                Alert("warning", "X", "Altro."),
            ),
            {**last, "status": "HALTED", "order_count": 0, "drawdown": -0.124},
            None,
        )
    )
    assert stop["title"].startswith("⛔")
    assert "Dal massimo -12,4%" in stop["body"] and "Sistema fermo" in stop["body"]
    assert "(+1 altri avvisi)" in stop["body"]
    assert stop["urgency"] == "high"


# ----------------------------------------------------------------------
# Server della web app
# ----------------------------------------------------------------------


@pytest.fixture()
def server(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO / "configs" / "paper_core_v1.toml", tmp_path / "configs")
    srv = create_server(tmp_path, "configs/paper_core_v1.toml", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()
    srv.server_close()


def _post(url, body, *, origin=None, content_type="application/json"):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    request.add_header("Content-Type", content_type)
    if origin:
        request.add_header("Origin", origin)
    return urllib.request.urlopen(request)


def test_pwa_files_are_served(server) -> None:
    base, _ = server
    for path, kind in (
        ("/manifest.webmanifest", "application/manifest+json"),
        ("/sw.js", "text/javascript"),
        ("/icons/icon-192.png", "image/png"),
        ("/icons/apple-touch-icon.png", "image/png"),
    ):
        with urllib.request.urlopen(base + path) as response:
            assert response.headers["Content-Type"].startswith(kind)
            assert response.read()

    with urllib.request.urlopen(base + "/") as response:
        html = response.read().decode()
    assert 'rel="manifest"' in html and "apple-touch-icon" in html


def test_subscribe_requires_same_origin_and_json(server) -> None:
    base, root = server
    _, _, _, subscription = _device()
    body = {"subscription": subscription, "label": "iPhone"}

    with pytest.raises(urllib.error.HTTPError) as error:
        _post(base + "/api/push/subscribe", body)
    assert error.value.code == 403

    with pytest.raises(urllib.error.HTTPError) as error:
        _post(base + "/api/push/subscribe", body, origin="https://evil.example.com")
    assert error.value.code == 403

    with pytest.raises(urllib.error.HTTPError) as error:
        _post(base + "/api/push/subscribe", body, origin=base, content_type="text/plain")
    assert error.value.code == 415

    with pytest.raises(urllib.error.HTTPError) as error:
        bad = {"subscription": {**subscription, "endpoint": "https://evil.example.com/x"}}
        _post(base + "/api/push/subscribe", bad, origin=base)
    assert error.value.code == 400

    with _post(base + "/api/push/subscribe", body, origin=base) as response:
        assert json.loads(response.read())["devices"] == 1
    assert len(SubscriptionStore(root / "data" / "push").all()) == 1

    with urllib.request.urlopen(base + "/api/push/key") as response:
        info = json.loads(response.read())
    assert info["devices"] == 1 and len(b64url_decode(info["publicKey"])) == 65

    with _post(base + "/api/push/unsubscribe", {"endpoint": APPLE}, origin=base) as response:
        assert json.loads(response.read())["devices"] == 0


def test_origin_rules() -> None:
    assert origin_allowed("http://127.0.0.1:8050", "127.0.0.1:8050")
    assert origin_allowed("https://pc.tail1234.ts.net", "pc.tail1234.ts.net")
    assert origin_allowed("https://pc.tail1234.ts.net", "127.0.0.1:8050")
    assert not origin_allowed("https://evil.example.com", "127.0.0.1:8050")
    assert not origin_allowed(None, "127.0.0.1:8050")

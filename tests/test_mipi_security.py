import builtins
import hashlib
import hmac
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from advanced_image_sensor_interface.sensor_interface.protocol.mipi import security


@pytest.fixture
def aesgcm():
    return pytest.importorskip("cryptography.hazmat.primitives.ciphers.aead").AESGCM


def proof_for(challenge, key=b"device-secret", method=security.AuthenticationMethod.HMAC_SHA256):
    digest = hashlib.sha512 if method == security.AuthenticationMethod.HMAC_SHA512 else hashlib.sha256
    return hmac.new(key, challenge, digest).digest()


def make_auth(**kwargs):
    auth = security.Authenticator(**kwargs)
    for identity in ("device", "other"):
        auth.register_credentials(security.SecurityCredentials(identity, pre_shared_key=b"device-secret"))
    return auth


def make_session(manager, identity="device"):
    manager.register_identity(security.SecurityCredentials(identity, pre_shared_key=b"device-secret"))
    challenge = manager.authenticator.generate_challenge(identity)
    session = manager.create_session(identity, proof_for(challenge), challenge)
    assert session is not None
    return session


def packet_for(aesgcm, session, counter, plaintext=b"frame"):
    nonce = counter.to_bytes(12, "big")
    sealed = aesgcm(session.session_key).encrypt(nonce, plaintext, None)
    return sealed[:-16], nonce, sealed[-16:]


@pytest.mark.parametrize(
    ("algorithm", "key_size", "ciphertext", "tag"),
    [
        (security.EncryptionAlgorithm.AES_128_GCM, 16, "0388dace60b6a392f328c2b971b2fe78", "ab6e47d42cec13bdf53a67b21257bddf"),
        (security.EncryptionAlgorithm.AES_256_GCM, 32, "cea7403d4d606b6e074ec5d3baf39d18", "d0d1c8a799996bf0265b98b5d48ab919"),
    ],
)
def test_nist_gcm_known_answer(aesgcm, algorithm, key_size, ciphertext, tag):
    protector = security.DataProtector(algorithm)
    protector.set_key(bytes(key_size))
    expected = bytes.fromhex(ciphertext), bytes.fromhex(tag)
    assert protector.encrypt(bytes(16), bytes(12)) == expected
    assert protector.decrypt(expected[0], bytes(12), expected[1]) == bytes(16)


@pytest.mark.parametrize(
    "algorithm,key_size", [(security.EncryptionAlgorithm.AES_128_GCM, 16), (security.EncryptionAlgorithm.AES_256_GCM, 32)]
)
@pytest.mark.parametrize("plaintext", [b"", b"frame" * 100])
def test_gcm_roundtrip_and_backend_interoperability(aesgcm, algorithm, key_size, plaintext):
    protector = security.DataProtector(algorithm)
    key, nonce = bytes(range(key_size)), bytes(range(12))
    protector.set_key(key)
    ciphertext, tag = protector.encrypt(plaintext, nonce)
    assert len(ciphertext) == len(plaintext)
    assert len(tag) == 16
    assert aesgcm(key).decrypt(nonce, ciphertext + tag, None) == plaintext
    assert protector.decrypt(ciphertext, nonce, tag) == plaintext


@pytest.mark.parametrize("component", ["ciphertext", "tag", "nonce", "key", "short_tag"])
def test_gcm_tampering_returns_none(aesgcm, component):
    protector = security.DataProtector()
    protector.set_key(bytes(16))
    nonce = bytes(12)
    ciphertext, tag = protector.encrypt(b"frame", nonce)
    if component == "ciphertext":
        ciphertext = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
    elif component == "tag":
        tag = bytes([tag[0] ^ 1]) + tag[1:]
    elif component == "nonce":
        nonce = bytes([1]) + nonce[1:]
    elif component == "key":
        protector.set_key(bytes([1]) * 16)
    else:
        tag = tag[:-1]
    assert protector.decrypt(ciphertext, nonce, tag) is None


@pytest.mark.parametrize(
    "algorithm,wrong_size", [(security.EncryptionAlgorithm.AES_128_GCM, 32), (security.EncryptionAlgorithm.AES_256_GCM, 16)]
)
def test_gcm_rejects_wrong_key_size(aesgcm, algorithm, wrong_size):
    protector = security.DataProtector(algorithm)
    with pytest.raises(ValueError, match=r"key|Key"):
        protector.set_key(bytes(wrong_size))


@pytest.mark.parametrize("nonce", [b"", bytes(11), bytes(13)])
def test_gcm_rejects_invalid_nonce(aesgcm, nonce):
    protector = security.DataProtector()
    protector.set_key(bytes(16))
    with pytest.raises(ValueError, match=r"nonce|Nonce"):
        protector.encrypt(b"frame", nonce)
    with pytest.raises(ValueError, match=r"nonce|Nonce"):
        protector.decrypt(b"frame", nonce, bytes(16))


def test_gcm_requires_key(aesgcm):
    protector = security.DataProtector()
    with pytest.raises(ValueError, match=r"key|Key"):
        protector.encrypt(b"frame", bytes(12))
    with pytest.raises(ValueError, match=r"key|Key"):
        protector.decrypt(b"frame", bytes(12), bytes(16))


@pytest.mark.parametrize("algorithm", [security.EncryptionAlgorithm.AES_128_CBC, security.EncryptionAlgorithm.AES_256_CBC])
def test_unimplemented_encryption_rejected_without_backend(algorithm):
    with pytest.raises(ValueError, match=r"unsupported|Unsupported|implemented"):
        security.DataProtector(algorithm)
    with pytest.raises(ValueError, match=r"unsupported|Unsupported|implemented"):
        security.SecurityConfig(encryption=algorithm)


def test_missing_backend_import_and_encryption_fail_closed(monkeypatch):
    original_import = builtins.__import__

    def without_crypto(name, *args, **kwargs):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("backend deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_crypto)
    spec = importlib.util.spec_from_file_location("mipi_security_without_backend", security.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.SecurityConfig()
    protector = module.DataProtector(module.EncryptionAlgorithm.NONE)
    assert protector.encrypt(b"frame", b"") == (b"frame", b"")
    assert protector.decrypt(b"frame", b"", b"") == b"frame"
    config = module.SecurityConfig(encryption=module.EncryptionAlgorithm.NONE, authentication=module.AuthenticationMethod.NONE)
    manager = module.MIPISecurityManager(config)
    manager.register_identity(module.SecurityCredentials("device"))
    session = manager.create_session("device", b"")
    assert session is not None
    packet = manager.protect_data(session.session_id, b"frame")
    assert manager.verify_and_decrypt(session.session_id, *packet) == b"frame"
    for algorithm in (module.EncryptionAlgorithm.AES_128_GCM, module.EncryptionAlgorithm.AES_256_GCM):
        with pytest.raises(ImportError, match=r"security|cryptography"):
            module.DataProtector(algorithm)
    with pytest.raises(ImportError, match=r"security|cryptography"):
        module.MIPISecurityManager()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encryption": security.EncryptionAlgorithm.AES_256_GCM, "key_size": 128},
        {"encryption": security.EncryptionAlgorithm.AES_128_GCM, "key_size": 256},
        {"nonce_size": 8},
        {"session_timeout_s": float("nan")},
        {"session_timeout_s": float("inf")},
        {"max_failed_attempts": 0},
        {"challenge_timeout_s": 0},
        {"max_pending_challenges": 0},
        {"max_sessions": 0},
        {"replay_window_size": 0},
        {"replay_window_size": 4097},
        {"max_packets_per_session": 0},
        {"max_packets_per_session": 2**32 + 1},
    ],
)
def test_invalid_security_config(kwargs):
    with pytest.raises(ValueError):
        security.SecurityConfig(**kwargs)


@pytest.mark.parametrize(
    "method",
    [
        security.AuthenticationMethod.PRE_SHARED_KEY,
        security.AuthenticationMethod.HMAC_SHA256,
        security.AuthenticationMethod.HMAC_SHA512,
    ],
)
def test_challenge_single_use_identity_bound_and_proof_bound(method):
    auth = make_auth(method=method, max_failures=10)
    supplied = b"client-chosen-challenge"
    assert auth.authenticate("device", proof_for(supplied, method=method), supplied) == (False, None)
    challenge = auth.generate_challenge("device")
    proof = proof_for(challenge, method=method)
    assert auth.authenticate("other", proof, challenge) == (False, None)
    assert auth.authenticate("device", proof, challenge) == (True, security.PrivilegeLevel.USER)
    assert auth.authenticate("device", proof, challenge) == (False, None)
    fresh = auth.generate_challenge("device")
    assert auth.authenticate("device", proof, fresh) == (False, None)
    assert auth.authenticate("device", proof_for(fresh, method=method), fresh) == (False, None)
    final = auth.generate_challenge("device")
    assert auth.authenticate("device", proof_for(final, method=method), final)[0]


def test_challenge_expiration_and_failed_attempt_lockout(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    auth = make_auth(max_failures=2, challenge_timeout_s=5)
    expired = auth.generate_challenge("device")
    now[0] += 5
    assert auth.authenticate("device", proof_for(expired), expired) == (False, None)
    assert auth._failure_counts["device"] == 1
    assert auth.authenticate("device", b"bad", None) == (False, None)
    assert auth._failure_counts["device"] == 2
    with pytest.raises(PermissionError, match="locked"):
        auth.generate_challenge("device")
    now[0] += 300
    fresh = auth.generate_challenge("device")
    assert auth.authenticate("device", proof_for(fresh), fresh)[0]
    assert auth._failure_counts["device"] == 0


def test_challenge_issuance_bounded_and_replacement_invalidates_old(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    auth = make_auth(max_pending_challenges=1, challenge_timeout_s=5)
    first = auth.generate_challenge("device")
    replacement = auth.generate_challenge("device")
    assert replacement != first
    with pytest.raises(RuntimeError, match="challenge"):
        auth.generate_challenge("other")
    assert len(auth._challenges) == 1
    assert auth.authenticate("device", proof_for(first), first) == (False, None)
    replacement = auth.generate_challenge("device")
    now[0] += 5
    assert auth.generate_challenge("other")
    assert len(auth._challenges) == 1
    for i in range(100):
        assert auth.authenticate(f"unknown-{i}", b"bad", b"bad") == (False, None)
    assert len(auth._failure_counts) == 2
    with pytest.raises(ValueError, match=r"identity|Identity"):
        auth.generate_challenge("unknown")
    with pytest.raises(ValueError, match=r"size|Size"):
        auth.generate_challenge("device", size=1)


def test_psk_never_accepts_raw_key():
    auth = make_auth(method=security.AuthenticationMethod.PRE_SHARED_KEY)
    challenge = auth.generate_challenge("device")
    assert auth.authenticate("device", b"device-secret", challenge) == (False, None)


def test_concurrent_challenge_replay_has_one_success():
    auth = make_auth(max_failures=20)
    challenge = auth.generate_challenge("device")
    barrier = Barrier(8)

    def authenticate(_):
        barrier.wait(timeout=5)
        return auth.authenticate("device", proof_for(challenge), challenge)[0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(authenticate, range(8))) == 1


def test_secret_free_repr():
    credentials = security.SecurityCredentials("device", pre_shared_key=b"SECRET_PSK", private_key_path="SECRET_PATH")
    session = security.Session("id", "device", security.PrivilegeLevel.USER, 0, 1, b"SECRET_SESSION_KEY")
    assert "SECRET" not in repr(credentials)
    assert "SECRET" not in repr(session)
    assert "device" in repr(credentials)
    assert "device" in repr(session)


def test_manager_tx_rx_separate_and_nonce_counters(aesgcm):
    manager = security.MIPISecurityManager()
    session = make_session(manager)
    for counter in range(3):
        packet = manager.protect_data(session.session_id, b"frame")
        assert packet is not None
        ciphertext, nonce, tag = packet
        assert nonce == counter.to_bytes(12, "big")
        assert len(tag) == 16
        assert aesgcm(session.session_key).decrypt(nonce, ciphertext + tag, None) == b"frame"
        assert manager.verify_and_decrypt(session.session_id, *packet) == b"frame"
        assert manager.verify_and_decrypt(session.session_id, *packet) is None
    assert session.nonce_counter == 3
    assert manager.get_metrics().packets_decrypted == 3


def test_bad_tag_and_wrong_session_do_not_consume_replay_state(aesgcm):
    manager = security.MIPISecurityManager()
    first, second = make_session(manager), make_session(manager, "other")
    ciphertext, nonce, tag = packet_for(aesgcm, first, 1000)
    assert manager.verify_and_decrypt(first.session_id, ciphertext, nonce, bytes(16)) is None
    low = packet_for(aesgcm, first, 0)
    assert manager.verify_and_decrypt(first.session_id, *low) == b"frame"
    assert manager.verify_and_decrypt(second.session_id, ciphertext, nonce, tag) is None
    assert manager.verify_and_decrypt(first.session_id, ciphertext, nonce, tag) == b"frame"
    assert manager.verify_and_decrypt(second.session_id, *packet_for(aesgcm, second, 1000)) == b"frame"
    assert manager.get_metrics().integrity_checks_failed == 2


def test_replay_window_rejects_stale_unseen_and_repeated_counters(aesgcm):
    manager = security.MIPISecurityManager(security.SecurityConfig(replay_window_size=8))
    session = make_session(manager)
    for counter in (2, 10, 9, 3):
        assert manager.verify_and_decrypt(session.session_id, *packet_for(aesgcm, session, counter)) == b"frame"
    for counter in (0, 1, 2, 3, 9, 10):
        assert manager.verify_and_decrypt(session.session_id, *packet_for(aesgcm, session, counter)) is None
    assert manager.verify_and_decrypt(session.session_id, *packet_for(aesgcm, session, 2**32 - 1)) == b"frame"
    assert session.receive_bitmap.bit_length() <= 8
    assert manager.verify_and_decrypt(session.session_id, *packet_for(aesgcm, session, 10)) is None
    assert not hasattr(manager, "_used_nonces")


def test_manager_replay_can_be_disabled_without_reusing_tx_nonces(aesgcm):
    manager = security.MIPISecurityManager(security.SecurityConfig(enable_replay_protection=False))
    session = make_session(manager)
    packet = manager.protect_data(session.session_id, b"frame")
    assert manager.verify_and_decrypt(session.session_id, *packet) == b"frame"
    assert manager.verify_and_decrypt(session.session_id, *packet) == b"frame"
    assert manager.protect_data(session.session_id, b"frame")[1] != packet[1]


def test_manager_key_isolation_during_interleaved_operations(aesgcm, monkeypatch):
    manager = security.MIPISecurityManager()
    first, second = make_session(manager), make_session(manager, "other")
    encrypt = security.DataProtector.encrypt
    interleaved = []

    def encrypt_interleaved(protector, plaintext, nonce):
        if plaintext == b"first":
            interleaved.append(manager.protect_data(second.session_id, b"second"))
        return encrypt(protector, plaintext, nonce)

    monkeypatch.setattr(security.DataProtector, "encrypt", encrypt_interleaved)
    first_packet = manager.protect_data(first.session_id, b"first")
    for session, packet, plaintext in ((first, first_packet, b"first"), (second, interleaved[0], b"second")):
        ciphertext, nonce, tag = packet
        assert aesgcm(session.session_key).decrypt(nonce, ciphertext + tag, None) == plaintext
        assert manager.verify_and_decrypt(session.session_id, *packet) == plaintext


def test_concurrent_sessions_and_duplicate_receive(aesgcm):
    manager = security.MIPISecurityManager()
    sessions = [make_session(manager, f"device-{i}") for i in range(4)]
    barrier = Barrier(8)

    def transfer(i):
        barrier.wait(timeout=5)
        session = sessions[i % 4]
        payload = str(i).encode()
        packet = manager.protect_data(session.session_id, payload)
        ciphertext, nonce, tag = packet
        assert aesgcm(session.session_key).decrypt(nonce, ciphertext + tag, None) == payload
        return session.session_id, nonce

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert len(set(pool.map(transfer, range(8)))) == 8
    session = sessions[0]
    packet = packet_for(aesgcm, session, 50)

    def receive(_):
        barrier.wait(timeout=5)
        return manager.verify_and_decrypt(session.session_id, *packet)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(receive, range(8)))
    assert results.count(b"frame") == 1
    assert results.count(None) == 7


def test_session_capacity_expiration_and_cleanup(aesgcm, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(security.time, "time", lambda: now[0])
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    manager = security.MIPISecurityManager(security.SecurityConfig(max_sessions=1, session_timeout_s=5))
    session = make_session(manager)
    assert manager.verify_and_decrypt(session.session_id, *packet_for(aesgcm, session, 0)) == b"frame"
    challenge = manager.authenticator.generate_challenge("device")
    assert manager.create_session("device", proof_for(challenge), challenge) is None
    assert len(manager._sessions) == 1
    assert len(manager.key_manager._session_keys) == 1
    now[0] += 5
    assert manager.protect_data(session.session_id, b"frame") is None
    assert not session.is_active
    assert session.session_key == b""
    assert session.receive_bitmap == 0
    assert manager.key_manager.get_session_key(session.session_id) is None
    assert manager.get_metrics().active_sessions == 0
    replacement = make_session(manager)
    now[0] += 5
    assert manager.cleanup_expired_sessions() == 1
    assert manager.validate_session(replacement.session_id) is None


def test_session_expiry_survives_wall_clock_rollback(aesgcm, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    manager = security.MIPISecurityManager(security.SecurityConfig(session_timeout_s=5))
    session = make_session(manager)
    monkeypatch.setattr(security.time, "time", lambda: session.created_at - 100)
    now[0] += 5
    assert manager.validate_session(session.session_id) is None


@pytest.mark.parametrize("counter", [2, 2**96])
def test_packet_limit_and_nonce_wrap_fail_explicitly(aesgcm, counter):
    manager = security.MIPISecurityManager(security.SecurityConfig(max_packets_per_session=2))
    session = make_session(manager)
    assert manager.protect_data(session.session_id, b"first") is not None
    assert manager.protect_data(session.session_id, b"last") is not None
    session.nonce_counter = counter
    with pytest.raises(OverflowError, match=r"nonce|Nonce|limit"):
        manager.protect_data(session.session_id, b"overflow")
    assert session.nonce_counter == counter
    assert manager.verify_and_decrypt(session.session_id, *packet_for(aesgcm, session, 2)) is None


def test_termination_rejects_packets_and_clears_state(aesgcm):
    manager = security.MIPISecurityManager()
    session = make_session(manager)
    packet = manager.protect_data(session.session_id, b"frame")
    assert manager.verify_and_decrypt(session.session_id, *packet) == b"frame"
    manager.terminate_session(session.session_id)
    assert manager.verify_and_decrypt(session.session_id, *packet) is None
    assert manager.protect_data(session.session_id, b"frame") is None
    assert session.session_key == b""
    assert session.nonce_counter == 0
    assert session.receive_bitmap == 0
    assert session.receive_highest == -1
    assert manager.get_metrics().active_sessions == 0

"""
MIPI Security Framework

Provides comprehensive security features for MIPI communication including
authentication, encryption, key management, and access control.

Key Features:
- AES-128/256 encryption for data protection
- HMAC-based message authentication
- Secure key exchange protocols
- Access control policies with privilege levels
- Session management with timeouts

Version: 4.0.0
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Optional

try:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    _HAS_CRYPTOGRAPHY = False
else:
    _HAS_CRYPTOGRAPHY = True

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for MIPI communication."""

    NONE = 0  # No security (development only)
    BASIC = 1  # Authentication only
    STANDARD = 2  # Authentication + integrity
    HIGH = 3  # Full encryption + authentication + integrity


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""

    NONE = "none"
    AES_128_CBC = "aes-128-cbc"
    AES_256_CBC = "aes-256-cbc"
    AES_128_GCM = "aes-128-gcm"
    AES_256_GCM = "aes-256-gcm"


class AuthenticationMethod(Enum):
    """Supported authentication methods."""

    NONE = "none"
    PRE_SHARED_KEY = "psk"
    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA512 = "hmac-sha512"
    CERTIFICATE = "certificate"


class PrivilegeLevel(Enum):
    """Access privilege levels."""

    GUEST = 0  # Read-only access
    USER = 1  # Basic operations
    OPERATOR = 2  # Operational control
    ADMIN = 3  # Full access


@dataclass
class SecurityConfig:
    """
    Configuration for MIPI security features.

    Attributes:
        security_level: Overall security level
        encryption: Encryption algorithm
        authentication: Authentication method
        key_size: Encryption key size in bits
        session_timeout_s: Session timeout in seconds
        max_failed_attempts: Max authentication failures before lockout
        enable_replay_protection: Enable replay attack protection
    """

    security_level: SecurityLevel = SecurityLevel.STANDARD
    encryption: EncryptionAlgorithm = EncryptionAlgorithm.AES_128_GCM
    authentication: AuthenticationMethod = AuthenticationMethod.HMAC_SHA256
    key_size: int = 128
    session_timeout_s: float = 3600.0  # 1 hour
    max_failed_attempts: int = 3
    enable_replay_protection: bool = True
    nonce_size: int = 12  # GCM nonce size in bytes
    challenge_timeout_s: float = 60.0
    max_pending_challenges: int = 1024
    max_sessions: int = 1024
    replay_window_size: int = 64
    max_packets_per_session: int = 2**32

    def __post_init__(self) -> None:
        """Validate and convert configuration values."""
        # Convert security_level string to enum if needed
        if isinstance(self.security_level, str):
            self.security_level = SecurityLevel[self.security_level.upper()]
        elif not isinstance(self.security_level, SecurityLevel):
            raise TypeError(f"security_level must be SecurityLevel enum or string, got {type(self.security_level).__name__}")

        # Convert encryption string to enum if needed
        if isinstance(self.encryption, str):
            self.encryption = EncryptionAlgorithm[self.encryption.upper()]

        # Convert authentication string to enum if needed
        if isinstance(self.authentication, str):
            self.authentication = AuthenticationMethod[self.authentication.upper()]

        if not isinstance(self.encryption, EncryptionAlgorithm):
            raise TypeError("encryption must be an EncryptionAlgorithm")
        if not isinstance(self.authentication, AuthenticationMethod):
            raise TypeError("authentication must be an AuthenticationMethod")
        if self.encryption not in (EncryptionAlgorithm.NONE, EncryptionAlgorithm.AES_128_GCM, EncryptionAlgorithm.AES_256_GCM):
            raise ValueError("Unsupported encryption algorithm")
        if self.key_size not in (128, 256):
            raise ValueError("Key size must be 128 or 256 bits")
        expected_size = 256 if self.encryption == EncryptionAlgorithm.AES_256_GCM else 128
        if self.encryption != EncryptionAlgorithm.NONE and self.key_size != expected_size:
            raise ValueError(f"Encryption algorithm requires a {expected_size}-bit key")
        if self.nonce_size != 12:
            raise ValueError("Nonce size must be 12 bytes")
        for name in ("session_timeout_s", "challenge_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "max_failed_attempts",
            "max_pending_challenges",
            "max_sessions",
            "replay_window_size",
            "max_packets_per_session",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.replay_window_size > 4096:
            raise ValueError("Replay window must not exceed 4096 packets")
        if self.max_packets_per_session > 2**32:
            raise ValueError("Session packet limit must not exceed 2**32")


@dataclass
class SecurityCredentials:
    """Security credentials for authentication."""

    identity: str
    privilege_level: PrivilegeLevel = PrivilegeLevel.USER
    pre_shared_key: Optional[bytes] = field(default=None, repr=False)
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate credentials."""
        if not self.identity:
            raise ValueError("Identity cannot be empty")


@dataclass
class Session:
    """Active security session."""

    session_id: str
    identity: str
    privilege_level: PrivilegeLevel
    created_at: float
    expires_at: float
    session_key: bytes = field(repr=False)
    nonce_counter: int = 0
    is_active: bool = True
    receive_highest: int = field(default=-1, init=False)
    receive_bitmap: int = field(default=0, init=False, repr=False)
    _expires_monotonic: float = field(default=0.0, init=False, repr=False)


@dataclass
class SecurityMetrics:
    """Security operation metrics."""

    authentications_successful: int = 0
    authentications_failed: int = 0
    packets_encrypted: int = 0
    packets_decrypted: int = 0
    integrity_checks_passed: int = 0
    integrity_checks_failed: int = 0
    active_sessions: int = 0


class KeyManager:
    """
    Manages cryptographic keys for MIPI security.

    Handles key generation, storage, rotation, and derivation.
    """

    def __init__(self, key_size: int = 128) -> None:
        """
        Initialize key manager.

        Args:
            key_size: Key size in bits (128 or 256)
        """
        self.key_size = key_size
        self.key_bytes = key_size // 8
        self._master_key: Optional[bytes] = None
        self._session_keys: dict[str, bytes] = {}

    def generate_master_key(self) -> bytes:
        """Generate a new master key."""
        self._master_key = secrets.token_bytes(self.key_bytes)
        logger.info("Generated new master key")
        return self._master_key

    def set_master_key(self, key: bytes) -> None:
        """Set the master key from external source."""
        if len(key) != self.key_bytes:
            raise ValueError(f"Key must be {self.key_bytes} bytes")
        self._master_key = key

    def derive_session_key(self, session_id: str, salt: bytes) -> bytes:
        """
        Derive a session key from the master key.

        Args:
            session_id: Session identifier
            salt: Random salt for key derivation

        Returns:
            Derived session key
        """
        if self._master_key is None:
            raise ValueError("Master key not set")

        # HKDF-like key derivation
        info = f"mipi_session_{session_id}".encode()
        prk = hmac.new(salt, self._master_key, hashlib.sha256).digest()
        okm = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()

        session_key = okm[: self.key_bytes]
        self._session_keys[session_id] = session_key
        return session_key

    def generate_nonce(self, size: int = 12) -> bytes:
        """Generate a random nonce."""
        return secrets.token_bytes(size)

    def get_session_key(self, session_id: str) -> Optional[bytes]:
        """Get session key by session ID."""
        return self._session_keys.get(session_id)

    def revoke_session_key(self, session_id: str) -> None:
        """Revoke a session key."""
        if session_id in self._session_keys:
            del self._session_keys[session_id]
            logger.info(f"Revoked session key for {session_id}")


class Authenticator:
    """
    Handles authentication for MIPI security.

    Supports multiple authentication methods including PSK and HMAC.
    """

    def __init__(
        self,
        method: AuthenticationMethod = AuthenticationMethod.HMAC_SHA256,
        max_failures: int = 3,
        challenge_timeout_s: float = 60.0,
        max_pending_challenges: int = 1024,
    ) -> None:
        """Initialize authentication with bounded, expiring challenge issuance."""
        if type(max_failures) is not int or max_failures <= 0:
            raise ValueError("max_failures must be a positive integer")
        if type(max_pending_challenges) is not int or max_pending_challenges <= 0:
            raise ValueError("max_pending_challenges must be a positive integer")
        if not math.isfinite(challenge_timeout_s) or challenge_timeout_s <= 0:
            raise ValueError("Challenge timeout must be finite and positive")
        self.method = method
        self.max_failures = max_failures
        self.challenge_timeout_s = challenge_timeout_s
        self.max_pending_challenges = max_pending_challenges
        self._credentials: dict[str, SecurityCredentials] = {}
        self._failure_counts: dict[str, int] = {}
        self._lockout_until: dict[str, float] = {}
        self._challenges: dict[str, tuple[bytes, float]] = {}
        self._lock = RLock()

    def register_credentials(self, credentials: SecurityCredentials) -> None:
        """Register credentials for an identity."""
        with self._lock:
            self._credentials[credentials.identity] = credentials
            self._failure_counts[credentials.identity] = 0
            self._lockout_until.pop(credentials.identity, None)
            self._challenges.pop(credentials.identity, None)
            logger.info(f"Registered credentials for {credentials.identity}")

    def _is_locked_out(self, identity: str, now: float) -> bool:
        until = self._lockout_until.get(identity)
        if until is None:
            return False
        if now < until:
            return True
        del self._lockout_until[identity]
        self._failure_counts[identity] = 0
        return False

    def authenticate(
        self, identity: str, proof: bytes, challenge: Optional[bytes] = None
    ) -> tuple[bool, Optional[PrivilegeLevel]]:
        """
        Authenticate an identity.

        Args:
            identity: Identity to authenticate
            proof: HMAC of the issued challenge (SHA256 for PSK)
            challenge: Server-issued challenge, required unless authentication is NONE

        Returns:
            Tuple of (success, privilege_level)
        """
        with self._lock:
            now = time.monotonic()
            credentials = self._credentials.get(identity)
            if credentials is None or self._is_locked_out(identity, now):
                return False, None

            pending = self._challenges.pop(identity, None)
            issued = (
                pending is not None and now < pending[1] and challenge is not None and hmac.compare_digest(challenge, pending[0])
            )
            is_valid = (self.method == AuthenticationMethod.NONE or issued) and self._verify_proof(credentials, proof, challenge)
            if is_valid:
                self._failure_counts[identity] = 0
                logger.info(f"Authentication successful for {identity}")
                return True, credentials.privilege_level

            self._failure_counts[identity] += 1
            if self._failure_counts[identity] >= self.max_failures:
                self._lockout_until[identity] = now + 300
                logger.warning(f"Identity {identity} locked out after {self.max_failures} failures")
            return False, None

    def _verify_proof(self, credentials: SecurityCredentials, proof: bytes, challenge: Optional[bytes]) -> bool:
        """Verify authentication proof."""
        if self.method == AuthenticationMethod.NONE:
            return True

        if self.method in (
            AuthenticationMethod.PRE_SHARED_KEY,
            AuthenticationMethod.HMAC_SHA256,
            AuthenticationMethod.HMAC_SHA512,
        ):
            if challenge is None or not credentials.pre_shared_key:
                return False

            hash_algo = hashlib.sha512 if self.method == AuthenticationMethod.HMAC_SHA512 else hashlib.sha256
            expected = hmac.new(credentials.pre_shared_key, challenge, hash_algo).digest()
            return hmac.compare_digest(proof, expected)

        return False

    def generate_challenge(self, identity: str, size: int = 32) -> bytes:
        """Issue one challenge per identity; PSK proofs are HMAC-SHA256(key, challenge)."""
        if type(size) is not int or not 16 <= size <= 64:
            raise ValueError("Challenge size must be between 16 and 64 bytes")
        with self._lock:
            if identity not in self._credentials:
                raise ValueError("Unknown identity")
            now = time.monotonic()
            if self._is_locked_out(identity, now):
                raise PermissionError("Identity is locked out")
            expired = [name for name, (_, deadline) in self._challenges.items() if now >= deadline]
            for name in expired:
                del self._challenges[name]
            if identity not in self._challenges and len(self._challenges) >= self.max_pending_challenges:
                raise RuntimeError("Pending challenge limit reached")
            challenge = secrets.token_bytes(size)
            self._challenges[identity] = challenge, now + self.challenge_timeout_s
            return challenge


class DataProtector:
    """
    Provides data protection (encryption/decryption) for MIPI communication.

    Uses symmetric encryption with authenticated encryption modes (GCM).
    """

    def __init__(self, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_128_GCM) -> None:
        """
        Initialize data protector.

        Args:
            algorithm: Encryption algorithm to use
        """
        if algorithm not in (EncryptionAlgorithm.NONE, EncryptionAlgorithm.AES_128_GCM, EncryptionAlgorithm.AES_256_GCM):
            raise ValueError("Unsupported encryption algorithm")
        if algorithm != EncryptionAlgorithm.NONE and not _HAS_CRYPTOGRAPHY:
            raise ImportError("AES-GCM requires cryptography; install advanced-image-sensor-interface[security]")
        self.algorithm = algorithm
        self._key: Optional[bytes] = None

    def set_key(self, key: bytes) -> None:
        """Set the encryption key."""
        if self.algorithm != EncryptionAlgorithm.NONE:
            key_size = 32 if self.algorithm == EncryptionAlgorithm.AES_256_GCM else 16
            if type(key) is not bytes or len(key) != key_size:
                raise ValueError(f"Key must be {key_size} bytes")
        self._key = key

    def _cipher(self, nonce: bytes) -> AESGCM:
        if not _HAS_CRYPTOGRAPHY:
            raise ImportError("AES-GCM requires cryptography; install advanced-image-sensor-interface[security]")
        if self._key is None:
            raise ValueError("Encryption key not set")
        if type(nonce) is not bytes or len(nonce) != 12:
            raise ValueError("Nonce must be 12 bytes")
        return AESGCM(self._key)

    def encrypt(self, plaintext: bytes, nonce: bytes) -> tuple[bytes, bytes]:
        """
        Encrypt data.

        Args:
            plaintext: Data to encrypt
            nonce: Unique nonce for this encryption

        Returns:
            Tuple of (ciphertext, authentication_tag)
        """
        if self.algorithm == EncryptionAlgorithm.NONE:
            return plaintext, b""

        sealed = self._cipher(nonce).encrypt(nonce, plaintext, None)
        return sealed[:-16], sealed[-16:]

    def decrypt(self, ciphertext: bytes, nonce: bytes, tag: bytes) -> Optional[bytes]:
        """
        Decrypt data.

        Args:
            ciphertext: Encrypted data
            nonce: Nonce used during encryption
            tag: Authentication tag

        Returns:
            Decrypted plaintext or None if authentication fails
        """
        if self.algorithm == EncryptionAlgorithm.NONE:
            return ciphertext

        cipher = self._cipher(nonce)
        if len(tag) != 16:
            return None
        try:
            return cipher.decrypt(nonce, ciphertext + tag, None)
        except InvalidTag:
            return None


class MIPISecurityManager:
    """
    High-level security manager for MIPI communication.

    Integrates key management, authentication, and data protection
    into a unified security framework.
    """

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        """
        Initialize security manager.

        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
        self.key_manager = KeyManager(self.config.key_size)
        self.authenticator = Authenticator(
            self.config.authentication,
            self.config.max_failed_attempts,
            self.config.challenge_timeout_s,
            self.config.max_pending_challenges,
        )
        self.data_protector = DataProtector(self.config.encryption)

        self._sessions: dict[str, Session] = {}
        self._metrics = SecurityMetrics()
        self._lock = RLock()

        # Generate master key
        self.key_manager.generate_master_key()

        logger.info(f"MIPI Security Manager initialized at level {self.config.security_level.name}")

    def register_identity(self, credentials: SecurityCredentials) -> None:
        """Register an identity for authentication."""
        self.authenticator.register_credentials(credentials)

    def create_session(self, identity: str, proof: bytes, challenge: Optional[bytes] = None) -> Optional[Session]:
        """
        Create a new authenticated session.

        Args:
            identity: Identity to authenticate
            proof: Authentication proof
            challenge: Optional challenge

        Returns:
            Session object if successful, None otherwise
        """
        with self._lock:
            success, privilege_level = self.authenticator.authenticate(identity, proof, challenge)
            if not success or privilege_level is None:
                self._metrics.authentications_failed += 1
                return None
            self._metrics.authentications_successful += 1
            self.cleanup_expired_sessions()
            if len(self._sessions) >= self.config.max_sessions:
                return None

            session_id = secrets.token_hex(16)
            salt = secrets.token_bytes(16)
            session_key = self.key_manager.derive_session_key(session_id, salt)
            now = time.time()
            session = Session(
                session_id=session_id,
                identity=identity,
                privilege_level=privilege_level,
                created_at=now,
                expires_at=now + self.config.session_timeout_s,
                session_key=session_key,
            )
            session._expires_monotonic = time.monotonic() + self.config.session_timeout_s
            self._sessions[session_id] = session
            self._metrics.active_sessions = len(self._sessions)
            logger.info(f"Created session {session_id} for {identity}")
            return session

    def validate_session(self, session_id: str) -> Optional[Session]:
        """
        Validate a session.

        Args:
            session_id: Session identifier

        Returns:
            Session if valid, None otherwise
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if not session.is_active or time.time() >= session.expires_at or time.monotonic() >= session._expires_monotonic:
                self.terminate_session(session_id)
                return None
            return session

    def terminate_session(self, session_id: str) -> None:
        """Terminate a session."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.is_active = False
                session.session_key = b""
                session.nonce_counter = 0
                session.receive_highest = -1
                session.receive_bitmap = 0
                self.key_manager.revoke_session_key(session_id)
                self._metrics.active_sessions = len(self._sessions)
                logger.info(f"Terminated session {session_id}")

    def protect_data(self, session_id: str, data: bytes) -> Optional[tuple[bytes, bytes, bytes]]:
        """
        Protect data with encryption and integrity.

        Args:
            session_id: Session identifier
            data: Data to protect

        Returns:
            Tuple of (ciphertext, nonce, tag) or None if session invalid
        """
        with self._lock:
            session = self.validate_session(session_id)
            if session is None:
                return None
            if not 0 <= session.nonce_counter < min(self.config.max_packets_per_session, 2**96):
                raise OverflowError("Session nonce limit reached; create a new session")
            nonce = session.nonce_counter.to_bytes(12, "big")
            session.nonce_counter += 1
            protector = DataProtector(self.config.encryption)
            protector.set_key(session.session_key)
            ciphertext, tag = protector.encrypt(data, nonce)
            self._metrics.packets_encrypted += 1
            return ciphertext, nonce, tag

    def verify_and_decrypt(self, session_id: str, ciphertext: bytes, nonce: bytes, tag: bytes) -> Optional[bytes]:
        """
        Verify integrity and decrypt data.

        Args:
            session_id: Session identifier
            ciphertext: Encrypted data
            nonce: Nonce used during encryption
            tag: Authentication tag

        Returns:
            Decrypted data or None if verification fails
        """
        with self._lock:
            session = self.validate_session(session_id)
            if session is None:
                return None
            if type(nonce) is not bytes or len(nonce) != 12:
                raise ValueError("Nonce must be 12 bytes")
            counter = int.from_bytes(nonce, "big")
            if counter >= self.config.max_packets_per_session:
                return None
            replay_protected = self.config.enable_replay_protection and self.config.encryption != EncryptionAlgorithm.NONE
            window = self.config.replay_window_size
            offset = session.receive_highest - counter
            if replay_protected and offset >= 0:
                if offset >= window or session.receive_bitmap & (1 << offset):
                    return None

            protector = DataProtector(self.config.encryption)
            protector.set_key(session.session_key)
            plaintext = protector.decrypt(ciphertext, nonce, tag)
            if plaintext is None:
                self._metrics.integrity_checks_failed += 1
                return None

            if replay_protected:
                if counter > session.receive_highest:
                    shift = counter - session.receive_highest
                    session.receive_bitmap = (
                        1 if shift >= window else ((session.receive_bitmap << shift) | 1) & ((1 << window) - 1)
                    )
                    session.receive_highest = counter
                else:
                    session.receive_bitmap |= 1 << offset
            self._metrics.integrity_checks_passed += 1
            self._metrics.packets_decrypted += 1
            return plaintext

    def check_privilege(self, session_id: str, required_level: PrivilegeLevel) -> bool:
        """
        Check if session has required privilege level.

        Args:
            session_id: Session identifier
            required_level: Required privilege level

        Returns:
            True if session has sufficient privileges
        """
        session = self.validate_session(session_id)
        if session is None:
            return False

        return session.privilege_level.value >= required_level.value

    def get_metrics(self) -> SecurityMetrics:
        """Get security metrics."""
        return self._metrics

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            now, monotonic_now = time.time(), time.monotonic()
            expired = [
                sid
                for sid, session in self._sessions.items()
                if not session.is_active or now >= session.expires_at or monotonic_now >= session._expires_monotonic
            ]
            for session_id in expired:
                self.terminate_session(session_id)
            return len(expired)

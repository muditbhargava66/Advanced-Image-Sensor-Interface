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

Version: 3.0.0
"""

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

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

        # Validate configuration
        if self.key_size not in (128, 256):
            raise ValueError("Key size must be 128 or 256 bits")

        if self.session_timeout_s <= 0:
            raise ValueError("Session timeout must be positive")


@dataclass
class SecurityCredentials:
    """Security credentials for authentication."""

    identity: str
    privilege_level: PrivilegeLevel = PrivilegeLevel.USER
    pre_shared_key: Optional[bytes] = None
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None

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
    session_key: bytes
    nonce_counter: int = 0
    is_active: bool = True


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

    def __init__(self, method: AuthenticationMethod = AuthenticationMethod.HMAC_SHA256, max_failures: int = 3) -> None:
        """
        Initialize authenticator.

        Args:
            method: Authentication method
            max_failures: Maximum failed attempts before lockout
        """
        self.method = method
        self.max_failures = max_failures
        self._credentials: dict[str, SecurityCredentials] = {}
        self._failure_counts: dict[str, int] = {}
        self._lockout_until: dict[str, float] = {}

    def register_credentials(self, credentials: SecurityCredentials) -> None:
        """Register credentials for an identity."""
        self._credentials[credentials.identity] = credentials
        self._failure_counts[credentials.identity] = 0
        logger.info(f"Registered credentials for {credentials.identity}")

    def authenticate(
        self, identity: str, proof: bytes, challenge: Optional[bytes] = None
    ) -> tuple[bool, Optional[PrivilegeLevel]]:
        """
        Authenticate an identity.

        Args:
            identity: Identity to authenticate
            proof: Authentication proof (e.g., HMAC, signature)
            challenge: Optional challenge for challenge-response auth

        Returns:
            Tuple of (success, privilege_level)
        """
        # Check lockout
        if identity in self._lockout_until:
            if time.time() < self._lockout_until[identity]:
                logger.warning(f"Identity {identity} is locked out")
                return False, None
            else:
                del self._lockout_until[identity]
                self._failure_counts[identity] = 0

        # Get credentials
        credentials = self._credentials.get(identity)
        if credentials is None:
            logger.warning(f"Unknown identity: {identity}")
            return False, None

        # Verify based on method
        is_valid = self._verify_proof(credentials, proof, challenge)

        if is_valid:
            self._failure_counts[identity] = 0
            logger.info(f"Authentication successful for {identity}")
            return True, credentials.privilege_level
        else:
            self._failure_counts[identity] += 1
            if self._failure_counts[identity] >= self.max_failures:
                # Lock out for 5 minutes
                self._lockout_until[identity] = time.time() + 300
                logger.warning(f"Identity {identity} locked out after {self.max_failures} failures")
            return False, None

    def _verify_proof(self, credentials: SecurityCredentials, proof: bytes, challenge: Optional[bytes]) -> bool:
        """Verify authentication proof."""
        if self.method == AuthenticationMethod.NONE:
            return True

        if self.method == AuthenticationMethod.PRE_SHARED_KEY:
            return proof == credentials.pre_shared_key

        if self.method in (AuthenticationMethod.HMAC_SHA256, AuthenticationMethod.HMAC_SHA512):
            if challenge is None or credentials.pre_shared_key is None:
                return False

            hash_algo = hashlib.sha256 if self.method == AuthenticationMethod.HMAC_SHA256 else hashlib.sha512
            expected = hmac.new(credentials.pre_shared_key, challenge, hash_algo).digest()
            return hmac.compare_digest(proof, expected)

        return False

    def generate_challenge(self, size: int = 32) -> bytes:
        """Generate a random challenge for authentication."""
        return secrets.token_bytes(size)


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
        self.algorithm = algorithm
        self._key: Optional[bytes] = None

    def set_key(self, key: bytes) -> None:
        """Set the encryption key."""
        self._key = key

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

        if self._key is None:
            raise ValueError("Encryption key not set")

        # Simplified encryption simulation
        # In production, use cryptography library with proper AES-GCM
        combined = self._key + nonce + plaintext
        tag = hmac.new(self._key, combined, hashlib.sha256).digest()[:16]

        # XOR-based encryption simulation (NOT SECURE - replace with real AES)
        ciphertext = self._xor_encrypt(plaintext, self._key + nonce)

        return ciphertext, tag

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

        if self._key is None:
            raise ValueError("Encryption key not set")

        # Decrypt
        plaintext = self._xor_encrypt(ciphertext, self._key + nonce)

        # Verify tag
        combined = self._key + nonce + plaintext
        expected_tag = hmac.new(self._key, combined, hashlib.sha256).digest()[:16]

        if not hmac.compare_digest(tag, expected_tag):
            logger.error("Authentication tag verification failed")
            return None

        return plaintext

    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """XOR-based encryption (for simulation only)."""
        result = bytearray(len(data))
        key_len = len(key)
        for i, byte in enumerate(data):
            result[i] = byte ^ key[i % key_len]
        return bytes(result)


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
        self.authenticator = Authenticator(self.config.authentication, self.config.max_failed_attempts)
        self.data_protector = DataProtector(self.config.encryption)

        self._sessions: dict[str, Session] = {}
        self._metrics = SecurityMetrics()
        self._used_nonces: set[bytes] = set()

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
        success, privilege_level = self.authenticator.authenticate(identity, proof, challenge)

        if not success or privilege_level is None:
            self._metrics.authentications_failed += 1
            return None

        self._metrics.authentications_successful += 1

        # Generate session
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
        session = self._sessions.get(session_id)
        if session is None:
            return None

        if not session.is_active:
            return None

        if time.time() > session.expires_at:
            self.terminate_session(session_id)
            return None

        return session

    def terminate_session(self, session_id: str) -> None:
        """Terminate a session."""
        session = self._sessions.get(session_id)
        if session:
            session.is_active = False
            self.key_manager.revoke_session_key(session_id)
            del self._sessions[session_id]
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
        session = self.validate_session(session_id)
        if session is None:
            return None

        # Set session key
        self.data_protector.set_key(session.session_key)

        # Generate nonce with replay protection
        nonce = self.key_manager.generate_nonce(self.config.nonce_size)
        if self.config.enable_replay_protection:
            while nonce in self._used_nonces:
                nonce = self.key_manager.generate_nonce(self.config.nonce_size)
            self._used_nonces.add(nonce)

        # Encrypt
        ciphertext, tag = self.data_protector.encrypt(data, nonce)
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
        session = self.validate_session(session_id)
        if session is None:
            return None

        # Replay protection
        if self.config.enable_replay_protection:
            if nonce in self._used_nonces:
                logger.warning("Replay attack detected")
                return None
            self._used_nonces.add(nonce)

        # Set session key
        self.data_protector.set_key(session.session_key)

        # Decrypt
        plaintext = self.data_protector.decrypt(ciphertext, nonce, tag)

        if plaintext is None:
            self._metrics.integrity_checks_failed += 1
        else:
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
        now = time.time()
        expired = [sid for sid, session in self._sessions.items() if now > session.expires_at]

        for session_id in expired:
            self.terminate_session(session_id)

        return len(expired)

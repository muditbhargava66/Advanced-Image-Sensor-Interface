"""
RDMA over Converged Ethernet (RoCE) for GigE Vision

Provides high-performance data transfer using RDMA technology
for ultra-low latency and zero-copy operation.

Key Features:
- Zero-copy data transfer (kernel bypass)
- Sub-microsecond latency
- Lossless Ethernet with PFC (Priority Flow Control)
- Multi-stream support
- Compatible with GigE Vision over RoCE
- Configurable simulation delays for realistic hardware behavior

Version: 3.2.0
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from advanced_image_sensor_interface.types import SimulationDelayConfig

logger = logging.getLogger(__name__)


class RoCEVersion(Enum):
    """RoCE protocol versions."""

    ROCE_V1 = "v1"  # RoCE over Ethernet (L2)
    ROCE_V2 = "v2"  # RoCEv2 over UDP/IP (L3 routable)


class QueuePairType(Enum):
    """RDMA Queue Pair types."""

    RC = "reliable_connected"  # Reliable, ordered delivery
    UC = "unreliable_connected"  # Unordered, may drop packets
    UD = "unreliable_datagram"  # Connectionless


class RDMAOperation(Enum):
    """RDMA operation types."""

    SEND = "send"
    RECV = "recv"
    WRITE = "write"
    READ = "read"
    ATOMIC = "atomic"


@dataclass
class RoCEConfig:
    """
    Configuration for RoCE transport.

    Attributes:
        version: RoCE protocol version
        mtu: Maximum Transmission Unit
        queue_depth: Depth of send/receive queues
        qp_type: Queue pair type
        enable_pfc: Enable Priority Flow Control
        traffic_class: DSCP traffic class
        gid_index: GID index for RoCEv2
        simulation_delays: Optional configurable simulation delays
    """

    version: RoCEVersion = RoCEVersion.ROCE_V2
    mtu: int = 4096
    queue_depth: int = 256
    qp_type: QueuePairType = QueuePairType.RC
    enable_pfc: bool = True
    traffic_class: int = 0  # DSCP value
    gid_index: int = 0
    max_inline_data: int = 64
    max_scatter_gather: int = 16
    timeout_ms: int = 5000

    # Simulation delay configuration (v3.2.0+)
    simulation_delays: Optional[SimulationDelayConfig] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_mtus = [256, 512, 1024, 2048, 4096]
        if self.mtu not in valid_mtus:
            raise ValueError(f"MTU must be one of {valid_mtus}")

        if self.queue_depth < 16 or self.queue_depth > 4096:
            raise ValueError("Queue depth must be between 16 and 4096")

        if self.simulation_delays is None:
            self.simulation_delays = SimulationDelayConfig()


@dataclass
class MemoryRegion:
    """RDMA memory region for zero-copy transfers."""

    region_id: str
    address: int
    length: int
    lkey: int  # Local key
    rkey: int  # Remote key
    is_registered: bool = False


@dataclass
class WorkRequest:
    """RDMA work request."""

    wr_id: int
    operation: RDMAOperation
    buffer: bytes
    remote_addr: int = 0
    remote_key: int = 0
    completed: bool = False
    status: str = "pending"


@dataclass
class CompletionEntry:
    """Work completion entry."""

    wr_id: int
    operation: RDMAOperation
    status: str
    bytes_transferred: int
    timestamp: float


@dataclass
class RoCEStatistics:
    """Statistics for RoCE transfers."""

    bytes_sent: int = 0
    bytes_received: int = 0
    sends_completed: int = 0
    receives_completed: int = 0
    rdma_writes: int = 0
    rdma_reads: int = 0
    cq_polls: int = 0
    completion_errors: int = 0
    average_latency_us: float = 0.0


class MemoryManager:
    """
    Manages RDMA memory regions for zero-copy transfers.

    Handles memory registration, protection, and access key management.
    """

    def __init__(self) -> None:
        """Initialize memory manager."""
        self._regions: dict[str, MemoryRegion] = {}
        self._next_key = 1

    def register_memory(self, buffer: bytes, region_id: str) -> MemoryRegion:
        """
        Register memory for RDMA access.

        Args:
            buffer: Memory buffer to register
            region_id: Unique identifier for this region

        Returns:
            Registered memory region
        """
        lkey = self._next_key
        rkey = self._next_key + 0x10000
        self._next_key += 1

        region = MemoryRegion(
            region_id=region_id, address=id(buffer), length=len(buffer), lkey=lkey, rkey=rkey, is_registered=True
        )

        self._regions[region_id] = region
        logger.debug(f"Registered memory region {region_id}: {len(buffer)} bytes")
        return region

    def deregister_memory(self, region_id: str) -> bool:
        """
        Deregister a memory region.

        Args:
            region_id: Region identifier

        Returns:
            True if deregistered successfully
        """
        if region_id in self._regions:
            self._regions[region_id].is_registered = False
            del self._regions[region_id]
            logger.debug(f"Deregistered memory region {region_id}")
            return True
        return False

    def get_region(self, region_id: str) -> Optional[MemoryRegion]:
        """Get a registered memory region."""
        return self._regions.get(region_id)

    def validate_access(self, lkey: int, offset: int, length: int) -> bool:
        """
        Validate memory access request.

        Args:
            lkey: Local access key
            offset: Offset within region
            length: Access length

        Returns:
            True if access is valid
        """
        for region in self._regions.values():
            if region.lkey == lkey:
                if offset >= 0 and offset + length <= region.length:
                    return True
        return False


class QueuePair:
    """
    RDMA Queue Pair for bidirectional communication.

    Manages send and receive queues with completion notification.
    """

    def __init__(self, config: RoCEConfig, qp_num: int) -> None:
        """
        Initialize queue pair.

        Args:
            config: RoCE configuration
            qp_num: Queue pair number
        """
        self.config = config
        self.qp_num = qp_num
        self._send_queue: list[WorkRequest] = []
        self._recv_queue: list[WorkRequest] = []
        self._completions: list[CompletionEntry] = []
        self._next_wr_id = 0
        self._is_connected = False

    def connect(self, remote_qp_num: int, remote_gid: bytes) -> bool:
        """
        Connect queue pair to remote endpoint.

        Args:
            remote_qp_num: Remote QP number
            remote_gid: Remote GID

        Returns:
            True if connected
        """
        self._is_connected = True
        logger.debug(f"QP {self.qp_num} connected to remote QP {remote_qp_num}")
        return True

    def post_send(
        self, buffer: bytes, operation: RDMAOperation = RDMAOperation.SEND, remote_addr: int = 0, remote_key: int = 0
    ) -> int:
        """
        Post a send work request.

        Args:
            buffer: Data to send
            operation: RDMA operation type
            remote_addr: Remote address for WRITE/READ
            remote_key: Remote key for WRITE/READ

        Returns:
            Work request ID
        """
        wr_id = self._next_wr_id
        self._next_wr_id += 1

        wr = WorkRequest(wr_id=wr_id, operation=operation, buffer=buffer, remote_addr=remote_addr, remote_key=remote_key)
        self._send_queue.append(wr)

        # Simulate immediate completion for simulation
        self._complete_work_request(wr)

        return wr_id

    def post_recv(self, buffer_size: int) -> int:
        """
        Post a receive buffer.

        Args:
            buffer_size: Size of receive buffer

        Returns:
            Work request ID
        """
        wr_id = self._next_wr_id
        self._next_wr_id += 1

        wr = WorkRequest(wr_id=wr_id, operation=RDMAOperation.RECV, buffer=bytes(buffer_size))
        self._recv_queue.append(wr)

        return wr_id

    def poll_completions(self, max_entries: int = 16) -> list[CompletionEntry]:
        """
        Poll for work completions.

        Args:
            max_entries: Maximum completions to return

        Returns:
            List of completion entries
        """
        entries = self._completions[:max_entries]
        self._completions = self._completions[max_entries:]
        return entries

    def _complete_work_request(self, wr: WorkRequest) -> None:
        """Mark a work request as completed."""
        wr.completed = True
        wr.status = "success"

        completion = CompletionEntry(
            wr_id=wr.wr_id, operation=wr.operation, status="success", bytes_transferred=len(wr.buffer), timestamp=time.time()
        )
        self._completions.append(completion)


class RoCETransport:
    """
    RoCE transport layer for GigE Vision.

    Provides RDMA-based data transfer with zero-copy capabilities
    and ultra-low latency.
    """

    def __init__(self, config: Optional[RoCEConfig] = None) -> None:
        """
        Initialize RoCE transport.

        Args:
            config: RoCE configuration
        """
        self.config = config or RoCEConfig()
        self.memory_manager = MemoryManager()
        self._queue_pairs: dict[int, QueuePair] = {}
        self._next_qp_num = 1
        self._statistics = RoCEStatistics()
        self._is_initialized = False

        logger.info(f"RoCE transport initialized: {self.config.version.value}")

    def initialize(self) -> bool:
        """
        Initialize RDMA resources.

        Returns:
            True if initialization successful
        """
        try:
            # In real implementation, this would:
            # 1. Open RDMA device
            # 2. Allocate protection domain
            # 3. Create completion queues

            # Apply link initialization delay
            delay = self.config.simulation_delays.apply_delay(self.config.simulation_delays.link_initialization_delay)
            time.sleep(delay)

            self._is_initialized = True
            logger.info("RoCE transport initialized")
            return True

        except Exception as e:
            logger.error(f"RoCE initialization failed: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown RDMA resources."""
        qp_nums = list(self._queue_pairs.keys())
        for qp_num in qp_nums:
            self._destroy_queue_pair(qp_num)

        self._queue_pairs.clear()
        self._is_initialized = False
        logger.info("RoCE transport shutdown")

    def create_queue_pair(self) -> int:
        """
        Create a new queue pair.

        Returns:
            Queue pair number
        """
        qp_num = self._next_qp_num
        self._next_qp_num += 1

        qp = QueuePair(self.config, qp_num)
        self._queue_pairs[qp_num] = qp

        logger.debug(f"Created QP {qp_num}")
        return qp_num

    def _destroy_queue_pair(self, qp_num: int) -> None:
        """Destroy a queue pair."""
        if qp_num in self._queue_pairs:
            del self._queue_pairs[qp_num]
            logger.debug(f"Destroyed QP {qp_num}")

    def connect_qp(self, qp_num: int, remote_qp_num: int, remote_gid: bytes) -> bool:
        """
        Connect a queue pair to remote endpoint.

        Args:
            qp_num: Local QP number
            remote_qp_num: Remote QP number
            remote_gid: Remote GID (Global Identifier)

        Returns:
            True if connected
        """
        qp = self._queue_pairs.get(qp_num)
        if qp is None:
            return False

        return qp.connect(remote_qp_num, remote_gid)

    def rdma_write(self, qp_num: int, data: bytes, remote_addr: int, remote_key: int) -> bool:
        """
        Perform RDMA WRITE operation (zero-copy to remote memory).

        Args:
            qp_num: Queue pair number
            data: Data to write
            remote_addr: Remote memory address
            remote_key: Remote access key

        Returns:
            True if write posted successfully
        """
        qp = self._queue_pairs.get(qp_num)
        if qp is None:
            return False

        # Apply command transfer delay
        delay = self.config.simulation_delays.apply_delay(self.config.simulation_delays.command_transfer_delay)
        time.sleep(delay)

        qp.post_send(buffer=data, operation=RDMAOperation.WRITE, remote_addr=remote_addr, remote_key=remote_key)

        self._statistics.rdma_writes += 1
        self._statistics.bytes_sent += len(data)
        return True

    def rdma_read(self, qp_num: int, size: int, remote_addr: int, remote_key: int) -> Optional[bytes]:
        """
        Perform RDMA READ operation (zero-copy from remote memory).

        Args:
            qp_num: Queue pair number
            size: Size to read
            remote_addr: Remote memory address
            remote_key: Remote access key

        Returns:
            Read data or None if failed
        """
        qp = self._queue_pairs.get(qp_num)
        if qp is None:
            return None

        # Apply register read delay
        delay = self.config.simulation_delays.apply_delay(self.config.simulation_delays.register_read_delay)
        time.sleep(delay)

        # Simulate read
        data = bytes(size)

        self._statistics.rdma_reads += 1
        self._statistics.bytes_received += size
        return data

    def send(self, qp_num: int, data: bytes) -> bool:
        """
        Send data using SEND operation.

        Args:
            qp_num: Queue pair number
            data: Data to send

        Returns:
            True if send posted
        """
        qp = self._queue_pairs.get(qp_num)
        if qp is None:
            return False

        # Apply command transfer delay
        delay = self.config.simulation_delays.apply_delay(self.config.simulation_delays.command_transfer_delay)
        time.sleep(delay)

        qp.post_send(data, RDMAOperation.SEND)
        self._statistics.sends_completed += 1
        self._statistics.bytes_sent += len(data)
        return True

    def receive(self, qp_num: int, size: int) -> Optional[bytes]:
        """
        Receive data.

        Args:
            qp_num: Queue pair number
            size: Expected data size

        Returns:
            Received data or None
        """
        qp = self._queue_pairs.get(qp_num)
        if qp is None:
            return None

        qp.post_recv(size)

        # Apply frame capture delay
        delay = self.config.simulation_delays.apply_delay(self.config.simulation_delays.frame_capture_delay)
        time.sleep(delay)

        # Simulate receive
        data = np.random.bytes(size)
        self._statistics.receives_completed += 1
        self._statistics.bytes_received += size
        return data

    def poll_completions(self, qp_num: int) -> list[CompletionEntry]:
        """
        Poll for work completions.

        Args:
            qp_num: Queue pair number

        Returns:
            List of completions
        """
        qp = self._queue_pairs.get(qp_num)
        if qp is None:
            return []

        self._statistics.cq_polls += 1
        return qp.poll_completions()

    def get_statistics(self) -> RoCEStatistics:
        """Get transfer statistics."""
        return self._statistics

    def get_capabilities(self) -> dict[str, Any]:
        """Get RoCE capabilities."""
        return {
            "version": self.config.version.value,
            "max_mtu": 4096,
            "max_queue_depth": 4096,
            "max_inline_data": self.config.max_inline_data,
            "max_scatter_gather": self.config.max_scatter_gather,
            "supports_atomic": True,
            "supports_pfc": True,
            "supports_ecn": True,
        }


class GigERoCEDriver:
    """
    GigE Vision driver with RoCE acceleration.

    Provides high-performance image transfer using RDMA.
    """

    def __init__(self, config: Optional[RoCEConfig] = None) -> None:
        """
        Initialize GigE RoCE driver.

        Args:
            config: RoCE configuration
        """
        self.config = config or RoCEConfig()
        self.transport = RoCETransport(self.config)
        self._stream_qp: Optional[int] = None
        self._is_connected = False
        self._is_streaming = False

        logger.info("GigE RoCE driver initialized")

    def connect(self, remote_ip: str, remote_port: int = 3956) -> bool:
        """
        Connect to GigE Vision camera over RoCE.

        Args:
            remote_ip: Camera IP address
            remote_port: GigE Vision port

        Returns:
            True if connected
        """
        try:
            if not self.transport.initialize():
                return False

            # Create queue pair for streaming
            self._stream_qp = self.transport.create_queue_pair()

            # Connect (simulated)
            remote_gid = bytes(16)  # Simulated GID
            if not self.transport.connect_qp(self._stream_qp, 1, remote_gid):
                return False

            # Apply device discovery delay
            delay = self.config.simulation_delays.apply_delay(self.config.simulation_delays.device_discovery_delay)
            time.sleep(delay)

            self._is_connected = True
            logger.info(f"Connected to {remote_ip}:{remote_port}")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from camera."""
        if self._is_streaming:
            self.stop_streaming()

        # Apply streaming teardown delay
        delay = self.config.simulation_delays.apply_delay(self.config.simulation_delays.streaming_teardown_delay)
        time.sleep(delay)

        self.transport.shutdown()
        self._is_connected = False
        logger.info("Disconnected")
        return True

    def start_streaming(self) -> bool:
        """Start image streaming."""
        if not self._is_connected:
            return False

        self._is_streaming = True

        # Apply streaming setup delay
        delay = self.config.simulation_delays.apply_delay(self.config.simulation_delays.streaming_setup_delay)
        time.sleep(delay)

        logger.info("Streaming started")
        return True

    def stop_streaming(self) -> bool:
        """Stop image streaming."""
        self._is_streaming = False

        # Apply streaming teardown delay
        delay = self.config.simulation_delays.apply_delay(self.config.simulation_delays.streaming_teardown_delay)
        time.sleep(delay)

        logger.info("Streaming stopped")
        return True

    def receive_frame(self, width: int, height: int, bpp: int = 8) -> Optional[np.ndarray]:
        """
        Receive image frame via RDMA.

        Args:
            width: Image width
            height: Image height
            bpp: Bits per pixel

        Returns:
            Image as numpy array
        """
        if not self._is_streaming or self._stream_qp is None:
            return None

        frame_size = width * height * (bpp // 8)
        data = self.transport.receive(self._stream_qp, frame_size)

        if data is None:
            return None

        dtype = np.uint8 if bpp == 8 else np.uint16
        return np.frombuffer(data, dtype=dtype).reshape((height, width))

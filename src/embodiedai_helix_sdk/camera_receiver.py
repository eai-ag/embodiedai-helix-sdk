"""Lightweight RTP JPEG receiver (RFC 2435) over UDP.

Only depends on Python stdlib. Receives JPEG frames from a GStreamer
``rtpjpegpay ! udpsink`` pipeline and reassembles them.
"""

import socket
import struct
import threading
from io import BytesIO
from typing import Optional, Callable

# RFC 2435 standard Luma/Chroma quantization tables (for Q 1-99)
_LUMA_QUANT = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

_CHROMA_QUANT = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]

# JPEG zigzag order
_ZIGZAG = [
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
]


def _make_quant_tables(q: int) -> tuple:
    """Generate quantization tables from Q factor (RFC 2435 Appendix A)."""
    if q < 1:
        q = 1
    elif q > 99:
        q = 99

    if q < 50:
        s = 5000 // q
    else:
        s = 200 - 2 * q

    luma = bytearray(64)
    chroma = bytearray(64)
    for i in range(64):
        lq = (_LUMA_QUANT[i] * s + 50) // 100
        cq = (_CHROMA_QUANT[i] * s + 50) // 100
        luma[_ZIGZAG[i]] = max(1, min(255, lq))
        chroma[_ZIGZAG[i]] = max(1, min(255, cq))

    return bytes(luma), bytes(chroma)


def _build_jpeg_header(width: int, height: int, type_: int, q: int,
                       luma_qt: bytes, chroma_qt: bytes) -> bytes:
    """Build a minimal JFIF header for the reassembled JPEG frame."""
    hdr = bytearray()

    # SOI
    hdr += b'\xff\xd8'

    # DQT - Luma
    hdr += b'\xff\xdb'
    hdr += struct.pack('>H', 67)  # length
    hdr += b'\x00'  # table 0, 8-bit precision
    hdr += luma_qt

    # DQT - Chroma
    hdr += b'\xff\xdb'
    hdr += struct.pack('>H', 67)
    hdr += b'\x01'  # table 1
    hdr += chroma_qt

    # SOF0
    hdr += b'\xff\xc0'
    nb_components = 3
    sof_len = 8 + 3 * nb_components
    hdr += struct.pack('>H', sof_len)
    hdr += b'\x08'  # 8-bit precision

    hdr += struct.pack('>HH', height, width)
    hdr += struct.pack('B', nb_components)

    # Component 1 (Y): sampling depends on type
    if type_ == 0:
        # 4:2:2
        hdr += b'\x01\x21\x00'
    else:
        # 4:2:0
        hdr += b'\x01\x22\x00'
    # Component 2 (Cb)
    hdr += b'\x02\x11\x01'
    # Component 3 (Cr)
    hdr += b'\x03\x11\x01'

    # DHT (Huffman tables) - standard tables
    hdr += _standard_huffman_tables()

    # SOS
    hdr += b'\xff\xda'
    sos_len = 6 + 2 * nb_components
    hdr += struct.pack('>H', sos_len)
    hdr += struct.pack('B', nb_components)
    hdr += b'\x01\x00'  # Y  -> DC table 0, AC table 0
    hdr += b'\x02\x11'  # Cb -> DC table 1, AC table 1
    hdr += b'\x03\x11'  # Cr -> DC table 1, AC table 1
    hdr += b'\x00\x3f\x00'  # Ss, Se, Ah/Al

    return bytes(hdr)


def _standard_huffman_tables() -> bytes:
    """Return standard JPEG Huffman tables (from JPEG spec Annex K)."""
    # DC Luma
    dc_luma_bits = bytes([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    dc_luma_vals = bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # DC Chroma
    dc_chroma_bits = bytes([0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    dc_chroma_vals = bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # AC Luma
    ac_luma_bits = bytes([0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d])
    ac_luma_vals = bytes([
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
        0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
        0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
        0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ])

    # AC Chroma
    ac_chroma_bits = bytes([0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77])
    ac_chroma_vals = bytes([
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
        0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
        0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
        0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ])

    tables = bytearray()
    for cls, bits, vals in [
        (0x00, dc_luma_bits, dc_luma_vals),
        (0x10, ac_luma_bits, ac_luma_vals),
        (0x01, dc_chroma_bits, dc_chroma_vals),
        (0x11, ac_chroma_bits, ac_chroma_vals),
    ]:
        tables += b'\xff\xc4'
        length = 2 + 1 + 16 + len(vals)
        tables += struct.pack('>H', length)
        tables += struct.pack('B', cls)
        tables += bits
        tables += vals

    return bytes(tables)


class RtpJpegReceiver:
    """Receives RTP JPEG (RFC 2435) frames over UDP.

    Usage::

        receiver = RtpJpegReceiver(host='10.99.2.70', port=5000)
        receiver.start()

        jpeg_bytes = receiver.get_frame()     # blocking, returns latest JPEG
        pil_image = receiver.get_image()       # returns PIL.Image (needs Pillow)

        receiver.stop()
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._latest_frame: Optional[bytes] = None

        # Reassembly state
        self._current_ts: Optional[int] = None
        self._fragments: dict = {}  # offset -> data

    def start(self):
        """Start receiving frames in a background thread."""
        if self._running:
            return

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self.host, self.port))

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the receiver."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._sock:
            self._sock.close()
            self._sock = None

    def get_frame(self, timeout: Optional[float] = 5.0) -> Optional[bytes]:
        """Block until a JPEG frame is available, return it as bytes."""
        if self._latest_frame is not None:
            with self._lock:
                return self._latest_frame

        self._frame_event.clear()
        if self._frame_event.wait(timeout=timeout):
            with self._lock:
                return self._latest_frame
        return None

    def get_image(self, timeout: Optional[float] = 5.0):
        """Block until a frame is available, return as PIL.Image."""
        from PIL import Image
        frame = self.get_frame(timeout=timeout)
        if frame is None:
            return None
        return Image.open(BytesIO(frame))

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < 12:
                continue

            self._process_rtp_packet(data)

    def _process_rtp_packet(self, data: bytes):
        # RTP header (12 bytes minimum)
        byte0, byte1 = data[0], data[1]
        marker = bool(byte1 & 0x80)
        pt = byte1 & 0x7F
        seq = struct.unpack('>H', data[2:4])[0]
        timestamp = struct.unpack('>I', data[4:8])[0]

        # Skip CSRC
        cc = byte0 & 0x0F
        hdr_len = 12 + cc * 4

        # Skip extension header if present
        if byte0 & 0x10:
            if len(data) < hdr_len + 4:
                return
            ext_len = struct.unpack('>H', data[hdr_len + 2:hdr_len + 4])[0]
            hdr_len += 4 + ext_len * 4

        if len(data) < hdr_len + 8:
            return

        payload = data[hdr_len:]

        # RFC 2435 JPEG header (8 bytes)
        # type-specific, fragment-offset (3 bytes), type, Q, width/8, height/8
        type_specific = payload[0]
        frag_offset = (payload[1] << 16) | (payload[2] << 8) | payload[3]
        type_ = payload[4]
        q = payload[5]
        width = payload[6] * 8
        height = payload[7] * 8

        pos = 8

        # Restart marker header (if type 64-127)
        if 64 <= type_ <= 127:
            if len(payload) < pos + 4:
                return
            pos += 4

        # Quantization table header (only in first fragment, Q >= 128)
        luma_qt = None
        chroma_qt = None
        if frag_offset == 0 and q >= 128:
            if len(payload) < pos + 4:
                return
            mbz, precision, qt_length = struct.unpack('>BBH', payload[pos:pos + 4])
            pos += 4
            if len(payload) < pos + qt_length:
                return
            qt_data = payload[pos:pos + qt_length]
            pos += qt_length
            if qt_length >= 128:
                luma_qt = qt_data[:64]
                chroma_qt = qt_data[64:128]
            elif qt_length >= 64:
                luma_qt = qt_data[:64]
                chroma_qt = qt_data[:64]

        jpeg_data = payload[pos:]

        # New frame?
        if self._current_ts != timestamp:
            self._current_ts = timestamp
            self._fragments = {}
            self._frame_width = width
            self._frame_height = height
            self._frame_type = type_ & 0x3F  # mask out restart bit
            self._frame_q = q
            self._frame_luma_qt = luma_qt
            self._frame_chroma_qt = chroma_qt

        self._fragments[frag_offset] = jpeg_data

        # Store quant tables from first fragment
        if frag_offset == 0 and luma_qt is not None:
            self._frame_luma_qt = luma_qt
            self._frame_chroma_qt = chroma_qt

        # If marker bit set, attempt reassembly
        if marker:
            self._reassemble_frame()

    def _reassemble_frame(self):
        # Sort fragments by offset and concatenate
        scan_data = bytearray()
        for offset in sorted(self._fragments.keys()):
            scan_data += self._fragments[offset]

        self._fragments = {}

        if not scan_data:
            return

        # Get quantization tables
        q = self._frame_q
        if q >= 128 and self._frame_luma_qt and self._frame_chroma_qt:
            luma_qt = self._frame_luma_qt
            chroma_qt = self._frame_chroma_qt
        elif q < 128:
            luma_qt, chroma_qt = _make_quant_tables(q)
        else:
            return  # no quant tables available

        header = _build_jpeg_header(
            self._frame_width, self._frame_height,
            self._frame_type, q, luma_qt, chroma_qt,
        )

        jpeg_frame = header + bytes(scan_data) + b'\xff\xd9'

        with self._lock:
            self._latest_frame = jpeg_frame
        self._frame_event.set()

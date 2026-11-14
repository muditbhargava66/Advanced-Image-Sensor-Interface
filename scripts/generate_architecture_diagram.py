#!/usr/bin/env python3
"""Generate the v4.0.0 layered architecture diagram as an SVG file.

The SVG is the canonical diagram source referenced by docs/system_architecture.md.
Convert it to PNG with CairoSVG, or with rsvg-convert or Inkscape:

    python scripts/generate_architecture_diagram.py
    uv run --no-project --with cairosvg cairosvg assets/system-architecture-v4.0.0.svg \
      -o assets/system-architecture-v4.0.0.png -s 2

Quick Look on macOS renders tall SVGs into a square thumbnail and crops the
lower layers, so it is not used for the committed PNG.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

WIDTH = 1600
MARGIN = 40
CARD_GAP = 14
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
INK = "#0f172a"
MUTED = "#475569"


def text(  # noqa: PLR0917
    x: float, y: float, value: str, size: int = 13, weight: int = 400, fill: str = INK, anchor: str = "start"
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def card(  # noqa: PLR0917
    x: float,
    y: float,
    width: float,
    height: float,
    accent: str,
    title: str,
    bullets: list[str],
    tag: str | None = None,
    dashed: bool = False,
) -> str:
    dash = ' stroke-dasharray="7 4"' if dashed else ""
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="10" '
        f'fill="#ffffff" fill-opacity="0.94" stroke="{accent}" stroke-width="1.6"{dash}/>',
        text(x + 16, y + 27, title, 15, 700, INK),
    ]
    if tag:
        tag_width = 7 * len(tag) + 16
        parts.append(
            f'<rect x="{x + width - tag_width - 12:.1f}" y="{y + 12:.1f}" width="{tag_width:.1f}" '
            f'height="20" rx="10" fill="{accent}" fill-opacity="0.14" stroke="{accent}" stroke-width="0.8"/>'
        )
        parts.append(text(x + width - tag_width / 2 - 12, y + 26, tag, 10, 600, accent, "middle"))
    for index, bullet in enumerate(bullets):
        parts.append(text(x + 16, y + 55 + index * 21, f"\u2022 {bullet}", 13, 400, MUTED))
    return "".join(parts)


def band(  # noqa: PLR0917
    y: float, height: float, code: str, title: str, note: str, fill: str, accent: str, cards: list[dict]
) -> str:
    parts = [
        f'<rect x="{MARGIN}" y="{y:.1f}" width="{WIDTH - 2 * MARGIN}" height="{height}" rx="14" '
        f'fill="{fill}" stroke="{accent}" stroke-width="1.2" stroke-opacity="0.45"/>',
        f'<rect x="{MARGIN + 16}" y="{y + 12:.1f}" width="48" height="24" rx="6" fill="{accent}"/>',
        text(MARGIN + 40, y + 29, code, 13, 700, "#ffffff", "middle"),
        text(MARGIN + 78, y + 30, title, 18, 700, INK),
        text(WIDTH - MARGIN - 18, y + 30, note, 12, 400, MUTED, "end"),
    ]
    count = len(cards)
    card_width = (WIDTH - 2 * MARGIN - 2 * 20 - (count - 1) * CARD_GAP) / count
    card_y = y + 48
    card_height = height - 64
    for index, item in enumerate(cards):
        card_x = MARGIN + 20 + index * (card_width + CARD_GAP)
        parts.append(
            card(
                card_x,
                card_y,
                card_width,
                card_height,
                accent,
                item["title"],
                item["bullets"],
                item.get("tag"),
                item.get("dashed", False),
            )
        )
    return "".join(parts)


def connector(y: float, label: str) -> str:
    return (
        f'<line x1="800" y1="{y + 2:.1f}" x2="800" y2="{y + 30:.1f}" stroke="#475569" '
        f'stroke-width="1.8" marker-end="url(#arrowhead)"/>' + text(814, y + 21, label, 12, 600, MUTED)
    )


def build_svg() -> str:
    layers = [
        {
            "code": "L0",
            "title": "Sensor input and declared formats",
            "note": "Simulation model; no hardware acquisition",
            "fill": "#fffbeb",
            "accent": "#d97706",
            "height": 190,
            "cards": [
                {
                    "title": "EnhancedSensorInterface",
                    "bullets": [
                        "VGA through 8K (7680 x 4320)",
                        "RAW8 / 10 / 12 / 14 / 16 / 20",
                        "HDR modes and Bayer patterns",
                        "Multi-sensor configuration",
                    ],
                },
                {
                    "title": "PatternGenerator",
                    "bullets": [
                        "color_bars, grayscale_ramp",
                        "checkerboard, slanted_edge",
                        "siemens_star, zone_plate, noise",
                        "Bit-depth aware output",
                    ],
                },
                {
                    "title": "Lens profiles",
                    "bullets": [
                        "Brown-Conrady radial + tangential",
                        "gopro_wide, smartphone_wide",
                        "dslr_kit, industrial_cs",
                        "Nearest and bilinear remap",
                    ],
                },
                {
                    "title": "Calibration data models",
                    "bullets": [
                        "CalibrationResult / StereoCalibrationResult",
                        "CalibrationDatabase store and export",
                        "Checkerboard and circles patterns",
                        "Quality metrics and RMS error",
                    ],
                },
            ],
        },
        {
            "code": "L1",
            "title": "Transport protocols (simulation)",
            "note": "Measured frame rates replace configured link capacity",
            "fill": "#eff6ff",
            "accent": "#2563eb",
            "height": 226,
            "cards": [
                {
                    "title": "MIPI CSI-2",
                    "bullets": [
                        "D-PHY v2.5 model, 1-4 lanes",
                        "Up to 4.5 Gbps per lane",
                        "ECC / CRC packet validation",
                        "Virtual channels 0-3",
                    ],
                },
                {
                    "title": "CoaXPress",
                    "bullets": [
                        "CXP-1 through CXP-12 grades",
                        "12.5 Gbps per link",
                        "PoCXP and trigger control",
                        "Register-map simulation",
                    ],
                },
                {
                    "title": "GigE Vision",
                    "bullets": [
                        "GVCP / GVSP, packet resend",
                        "1 / 2.5 / 5 / 10 GigE",
                        "Optional NumPy-backed RoCE",
                        "No hardware RDMA",
                    ],
                },
                {
                    "title": "USB3 Vision",
                    "bullets": [
                        "SuperSpeed and SuperSpeedPlus",
                        "Locked FIFO buffer ownership",
                        "Discovery and hot-plug model",
                        "GenICam feature nodes",
                    ],
                },
            ],
        },
        {
            "code": "L2",
            "title": "Frame path, integrity, and security",
            "note": "Application envelope is not transport verification",
            "fill": "#fff1f2",
            "accent": "#e11d48",
            "height": 238,
            "cards": [
                {
                    "title": "FrameCodec",
                    "bullets": [
                        "Explicit ImageFormat declaration",
                        "Little-endian unpacked samples",
                        "8/10/12/14/16/20-bit; owned arrays",
                        "Rejects float64 and packed formats",
                    ],
                },
                {
                    "title": "FramePipeline",
                    "bullets": [
                        "Capture, decode, then typed operation",
                        "Measured latency and throughput",
                        "Failures propagate as typed results",
                        "Sequential stereo pair -> DepthResult",
                    ],
                },
                {
                    "title": "IntegrityChecker",
                    "bullets": [
                        "CRC-32 mismatch fails closed",
                        "Hamming candidates re-checked by CRC",
                        "REED_SOLOMON is detection only",
                        "Opt-in sender-side envelope",
                    ],
                },
                {
                    "title": "MIPI Security",
                    "bullets": [
                        "AES-128/256-GCM, 12-byte nonce",
                        "PSK / HMAC challenge, single use",
                        "Per-session TX counters",
                        "Authenticated RX replay window",
                    ],
                    "tag": "optional [security]",
                    "dashed": True,
                },
            ],
        },
        {
            "code": "L3",
            "title": "Typed processing pipelines",
            "note": "All processors return explicit success/error/metrics",
            "fill": "#f0fdf4",
            "accent": "#16a34a",
            "height": 226,
            "cards": [
                {
                    "title": "SignalProcessor",
                    "bullets": [
                        "Noise reduction and color correction",
                        "8/10/12/14/16/20-bit integer input",
                        "Rejects float64 and non-finite data",
                        "SignalProcessingResult",
                    ],
                },
                {
                    "title": "HDRProcessor",
                    "bullets": [
                        "Reinhard, Drago, Adaptive, Gamma",
                        "Mertens and weighted-average fusion",
                        "Mantiuk/Durand enums fall back",
                        "HDRProcessingResult",
                    ],
                },
                {
                    "title": "RAWProcessor",
                    "bullets": [
                        "Bilinear, Malvar, and Simple",
                        "RGGB / BGGR / GRBG / GBRG",
                        "AHD/VNG enums fall back to Simple",
                        "RAWProcessingResult",
                    ],
                },
                {
                    "title": "LensCorrectionPipeline",
                    "bullets": [
                        "Brown-Conrady distortion model",
                        "Radial and tangential terms",
                        "Nearest and bilinear interpolation",
                        "LensCorrectionResult",
                    ],
                },
            ],
        },
        {
            "code": "L4",
            "title": "Depth and calibration",
            "note": "Optional numba and trimesh acceleration",
            "fill": "#faf5ff",
            "accent": "#7c3aed",
            "height": 214,
            "cards": [
                {
                    "title": "StereoDepthProcessor",
                    "bullets": [
                        "Block matching and 4/8-path SGM",
                        "Numba kernels or numpy fallback",
                        "Disparity -> depth -> point cloud",
                        "DepthResult and PLY export",
                    ],
                },
                {
                    "title": "Calibration solvers",
                    "bullets": [
                        "OpenCV chessboard detection",
                        "Native Zhang method (numpy/scipy)",
                        "DLT + RQ projection matrix",
                        "prefer_native_calibration switch",
                    ],
                },
                {
                    "title": "NeuralCalibrationTuner",
                    "bullets": [
                        "scikit-learn MLPRegressor",
                        "Sobel / Harris / FFT / Laplacian features",
                        "Quality prediction and parameter tuning",
                        "No TensorFlow or PyTorch dependency",
                    ],
                },
            ],
        },
        {
            "code": "L5",
            "title": "Platform services (cross-cutting)",
            "note": "Available to every layer above",
            "fill": "#f0f9ff",
            "accent": "#0284c7",
            "height": 222,
            "cards": [
                {
                    "title": "Power management",
                    "bullets": [
                        "7 states and 5 operating modes",
                        "Thermal states and DVFS model",
                        "Budgets and array-level control",
                        "Simulation and hardware backends",
                    ],
                },
                {
                    "title": "Performance",
                    "bullets": [
                        "Profiler and system monitor",
                        "LRU + TTL cache manager",
                        "Automatic parameter tuning",
                        "SNR, dynamic range, color accuracy",
                    ],
                },
                {
                    "title": "Buffers",
                    "bullets": [
                        "BufferManager and AsyncBufferManager",
                        "ManagedBuffer context manager",
                        "DMA, memory-mapped, zero-copy models",
                        "Pool statistics and resizing",
                    ],
                },
                {
                    "title": "Errors and config",
                    "bullets": [
                        "SensorError exception hierarchy",
                        "Circuit breaker and retry policy",
                        "Environment-aware ConfigManager",
                        "Thread-safe lazy singleton",
                    ],
                },
            ],
        },
        {
            "code": "L6",
            "title": "Outputs and verification",
            "note": "Local gates; hosted CI has not run for this branch",
            "fill": "#f8fafc",
            "accent": "#475569",
            "height": 196,
            "cards": [
                {
                    "title": "Outputs and metrics",
                    "bullets": [
                        "Processed frames and depth maps",
                        "PLY point clouds and mesh export",
                        "SNR, dynamic range, Delta E",
                        "PNG visualizations",
                    ],
                },
                {
                    "title": "Quality gates",
                    "bullets": [
                        "640 passed, 646 collected, 6 skipped",
                        "ruff, black, mypy, pyright",
                        "bandit and dependency audits",
                        "Sphinx documentation build",
                    ],
                },
                {
                    "title": "Examples and scripts",
                    "bullets": [
                        "9 Python examples + notebook",
                        "simulation.py, automated_testing.py",
                        "data_analysis.py with plot/compare",
                        "benchmarks and stubs",
                    ],
                },
            ],
        },
    ]

    connectors = [
        "raw sensor frames",
        "unpacked payload bytes",
        "validated arrays",
        "typed results",
        "depth and calibration",
        "measured metrics",
    ]

    height = 120 + 20
    for layer in layers:
        height += layer["height"] + 34
    extensions_y = height + 4
    height = extensions_y + 92 + 74

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Advanced Image Sensor Interface v4.0.0 system architecture</title>',
        '<desc id="desc">Layered architecture: sensor input, transport protocols, frame codec and pipeline, '
        "integrity and AES-GCM security, typed processing, depth and calibration, platform services, outputs, "
        "and verification. Protocol drivers and hardware control are simulations.</desc>",
        "<defs>",
        '<linearGradient id="header" x1="0%" y1="0%" x2="100%" y2="0%">',
        '<stop offset="0%" stop-color="#0f172a"/>',
        '<stop offset="100%" stop-color="#1d4ed8"/>',
        "</linearGradient>",
        '<marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">',
        '<polygon points="0 0, 10 4, 0 8" fill="#475569"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{WIDTH}" height="{height}" fill="#f8fafc"/>',
        f'<rect width="{WIDTH}" height="120" fill="url(#header)"/>',
        text(WIDTH / 2, 54, "Advanced Image Sensor Interface v4.0.0", 30, 700, "#ffffff", "middle"),
        text(
            WIDTH / 2,
            84,
            "Simulation framework  |  MIPI CSI-2  |  CoaXPress CXP-12  |  GigE Vision  |  USB3 Vision  |  AES-GCM",
            14,
            400,
            "#dbeafe",
            "middle",
        ),
        f'<rect x="{WIDTH - 146}" y="30" width="106" height="28" rx="14" fill="#ffffff" fill-opacity="0.14" '
        f'stroke="#ffffff" stroke-opacity="0.45"/>',
        text(WIDTH - 93, 49, "v4.0.0", 13, 700, "#ffffff", "middle"),
    ]

    y = 140
    for index, layer in enumerate(layers):
        parts.append(
            band(y, layer["height"], layer["code"], layer["title"], layer["note"], layer["fill"], layer["accent"], layer["cards"])
        )
        y += layer["height"]
        if index < len(connectors):
            parts.append(connector(y, connectors[index]))
            y += 34

    parts.append(
        f'<rect x="{MARGIN}" y="{extensions_y:.1f}" width="{WIDTH - 2 * MARGIN}" height="92" rx="12" '
        f'fill="#ffffff" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="7 4"/>'
    )
    parts.append(
        text(MARGIN + 20, extensions_y + 27, "Example extensions (not importable from the installed package)", 15, 700, INK)
    )
    parts.append(
        text(
            MARGIN + 20,
            extensions_y + 52,
            "AIEnhancedProcessor, SceneClassifier, NoisePredictor, QualityAssessor  |  examples/ai_ml_enhancements.py",
            13,
            400,
            MUTED,
        )
    )
    parts.append(
        text(
            MARGIN + 20,
            extensions_y + 74,
            "AINoiseReducer, AdaptiveColorCorrector, HighSpeedMIPIDriver  |  examples/custom_extension.py",
            13,
            400,
            MUTED,
        )
    )

    legend_y = extensions_y + 118
    legend = [
        ("#ffffff", "#475569", "", "Core module"),
        ("#ffffff", "#e11d48", ' stroke-dasharray="7 4"', "Optional extra"),
        ("#f8fafc", "#94a3b8", ' stroke-dasharray="2 3"', "Simulation-only model"),
    ]
    legend_x = MARGIN + 18
    for fill, stroke, dash, label in legend:
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 10}" width="18" height="12" rx="3" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.4"{dash}/>'
        )
        parts.append(text(legend_x + 26, legend_y, label, 12, 500, MUTED))
        legend_x += 210
    parts.append(
        text(
            WIDTH - MARGIN - 18,
            legend_y,
            "Generated 2026-10-26 from source evidence  |  docs/system_architecture.md",
            12,
            400,
            MUTED,
            "end",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the v4.0.0 architecture SVG")
    parser.add_argument(
        "--output",
        default="assets/system-architecture-v4.0.0.svg",
        help="Output SVG path (default: assets/system-architecture-v4.0.0.svg)",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

# Qwen3-TTS 0.6B for Apple Neural Engine

Quality-verified, streaming English TTS for Apple Silicon. This repository
contains the Core ML conversion, ANE validation, benchmarking and release
tooling behind
[`erjigit17/Qwen3-TTS-0.6B-ANE`](https://huggingface.co/erjigit17/Qwen3-TTS-0.6B-ANE).
Model binaries live on Hugging Face; GitHub contains the reproducible source and
engineering evidence.

## Verified release

- 47.75 ms warm p95 from a prepared prompt to the first in-process PCM chunk
  across 300 measured runs on a 32 GB M4 MacBook Air.
- 1.418x real-time median throughput for the 4.08-second reference utterance.
- Five stability texts spanning 51–244 audio frames produced byte-identical PCM
  between the LUT6 dual-prefill path and the accepted FP16 Core ML path.
- Every reported neural operation in the shipped graphs passed the strict
  ANE-preferred compute-plan gate.
- No generated-audio cache is used.

These boundaries matter: model loading, Core ML compilation, fresh text
preparation, transport and audible playback onset are excluded from the
47.75 ms number. See [`BENCHMARKS.json`](BENCHMARKS.json) for structured facts
and [`MODEL_CARD.md`](MODEL_CARD.md) for the complete release statement.

## Use the released model

```bash
python3.12 -m pip install --upgrade huggingface_hub
hf download erjigit17/Qwen3-TTS-0.6B-ANE --local-dir Qwen3-TTS-0.6B-ANE
cd Qwen3-TTS-0.6B-ANE
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python verify_install.py --checksums
python verify_reference.py
python example.py "I'm sorry about the charge. I'll fix it for you." --output answer.wav
```

The Hugging Face folder is the supported consumption boundary. It contains the
runtime, tokenizer/frontend assets, Core ML graphs and sample. This GitHub
repository is for rebuilding, validating and improving the ANE port.

## Develop the ANE port

Read these in order:

1. [`docs/ANE_ENGINEERING_GUIDE.md`](docs/ANE_ENGINEERING_GUIDE.md) — reusable
   principles learned while porting transformer TTS to ANE.
2. [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) — commands, measurements,
   accepted candidates and failed experiments in chronological context.
3. [`MODEL_CARD.md`](MODEL_CARD.md) — claims that are safe to publish.

The conversion environment requires Apple Silicon, macOS 15+, Xcode command-line
tools and Python 3.12. The shipped runtime supports Python 3.10–3.13.

```bash
python3.12 -m venv qwen-env
qwen-env/bin/pip install -r requirements-qwen.txt
xcrun swiftc -parse-as-library ane_gate.swift -O -o /tmp/ane-gate
qwen-env/bin/python -m unittest discover -p 'test_*.py'
```

Weights, compiled Core ML caches, virtual environments, WAV files and generated
reports are deliberately excluded from Git. Exporters refuse to overwrite an
existing candidate so experimental evidence cannot be silently replaced.

## Repository map

- `export_*.py`, `quantize_*.py` — reproducible Core ML conversion experiments.
- `verify_*.py`, `check_*.py`, `test_*.py` — parity, stability and contract gates.
- `benchmark_*.py`, `measure_*.py` — scoped performance measurement.
- `voice_stream.py` — research runtime used to validate the graph composition.
- `build_hf_release.py`, `publish_hf_release.py` — deterministic HF packaging.
- `ane_gate.swift` — strict Core ML compute-plan admission check.
- `docs/` — durable ANE engineering knowledge and the experiment log.

## Project status

This is an unofficial Qwen3-TTS port and a developer preview, not an official
Qwen or Apple release. The current package exposes the English Serena voice and
does not claim reliable instruction/emotion control. See the Apache-2.0 license
and upstream model terms.

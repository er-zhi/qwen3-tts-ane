# Engineering Neural Models for Apple Neural Engine

This guide records the reusable lessons from converting Qwen3-TTS 0.6B into a
low-latency Core ML pipeline. It separates measured facts from assumptions so a
future ANE port can repeat the successful workflow without repeating every
failed experiment.

## 1. Treat ANE as a compiler target, not a Python device

Applications do not dispatch arbitrary PyTorch or NumPy work directly to ANE.
The supported boundary is a Core ML model. Core ML compiles the graph and chooses
execution backends according to graph support, shapes, data types, deployment
target and `MLComputeUnits`.

`CPU_AND_NE` excludes the GPU but does not mean that every operation executes on
ANE. Tokenization, orchestration, array construction, validation and PCM
serialization remain host work. Even a Core ML compute plan reports preferred
placement rather than a hardware execution trace. State claims precisely:

- **ANE-preferred neural operations** when the compute plan supports it;
- **CPU + ANE application** for the complete runtime;
- never “100% ANE” unless hardware tracing can prove the whole boundary.

The maintained `ane_gate.swift` fails a candidate when any reported neural
operation prefers CPU or GPU. Use it as an admission gate, then measure the real
device separately.

## 2. Decompose by latency boundary

An autoregressive TTS request contains different workloads:

1. text and speaker preparation;
2. talker prefill;
3. semantic-token selection;
4. residual code prediction;
5. waveform decoding;
6. continuation talker steps;
7. transport and playback.

Do not optimize “the model” as one number. Measure each stage around the first
PCM boundary and around steady-state frames. In this port the residual predictor
was the dominant first-frame cost; optimizing KV alone could never satisfy a
30 ms total budget while the predictor consumed approximately that budget.

Record cold, post-idle and warm measurements separately. Core ML compilation,
model loading, text preparation, first PCM, first audible sample and complete
file latency are different metrics. A benchmark must name what it includes.

## 3. Export fixed, explicit contracts

ANE performs best when the compiler can see stable shapes and supported tensor
operations. Prefer:

- fixed maximum capacities with masks;
- FP16 activations unless a verified integer path is faster;
- one clearly defined tensor layout at every component boundary;
- explicit state when recurrent history must survive;
- small graphs that can be independently admitted and benchmarked.

Validate input/output names, ranks and shapes when loading a graph. Reject an
incompatible package before inference rather than relying on a later native
failure. Bound text length, frame count, concurrency and output size.

## 4. KV state is correctness-sensitive

KV optimization can reduce repeated prefill, but it can also silently change
generation. The safe sequence is:

1. prove the invariant prefix token by token;
2. snapshot an immutable prefix state;
3. restore it before every request, including after cancellation;
4. mask future positions;
5. test migration when moving from a short cache to a larger cache;
6. compare complete codes, PCM and EOS against a fresh-prefill reference;
7. serialize access unless independent Core ML state objects are proven safe.

In this project a 9-token language/speaker prefix could be reused exactly, but
its end-to-end speedup was small and timing variance was material. KV reuse is a
production option, not permission to skip fresh-prefill measurements.

## 5. Quantize progressively and selectively

Apple's Core ML Tools guidance makes 8-bit weight-only quantization and 6/8-bit
palettization the safest data-free starting points. Four-bit compression usually
requires finer granularity, calibration or fine-tuning. The practical ladder is:

1. establish an accepted FP16 Core ML reference;
2. try weight-only INT8 per channel or k-means LUT8/LUT6;
3. quantify sensitivity per component and, where possible, per layer;
4. retain outliers, embeddings, normalization and sensitive projections in FP16;
5. for 4-bit, test grouped-channel palettization or per-block quantization;
6. if post-training compression loses quality, use calibration, GPTQ, SKM,
   DKM/QAT or fine-tuning before lowering precision further;
7. re-run placement, latency, memory and quality checks on the target chip.

Compression is not automatically acceleration. A backend may decompress weights
ahead of time, generate an unsupported graph, or add enough LUT/scale overhead to
lose the speed benefit. In one rejected experiment an 8-bit grouped-palettized
package expanded to about 7 GB and the compute plan reported no ANE-preferred
neural operations.

### Results from this port

- LUT6 k-means post-training palettization of the first-frame-only prefill graph
  was accepted. Combined with dual prefill, warm first PCM improved from
  64.39/69.67 ms p50/p95 to 44.88/47.75 ms.
- The startup prefill stage fell from about 25.50 ms to 7.40 ms, although this
  improvement also includes the reduced first-frame-only output contract and
  must not be attributed to quantization alone.
- Five texts spanning 51–244 frames matched the full FP16 prefill path exactly
  in generated codes and PCM.
- LUT4 changed 15 of 16 first-frame codes and was rejected.
- A broader Q8 talker candidate changed perceived voice character and was
  rejected even though it was fast.

“Exact” here describes the tested output boundary, not mathematical equality of
the compressed weights or every intermediate tensor.

## 6. Move work off the critical path before fusing everything

The accepted dual-prefill design uses two graphs in one request:

1. a compact LUT6 graph emits only the logits and hidden state needed for the
   first audio frame;
2. after the first PCM chunk is yielded, the full FP16 prefill reconstructs the
   exact continuation KV state.

This reduced time to first PCM without caching audio and preserved the complete
waveform. It did increase the first-to-second-chunk interval, so sustained
streaming must be measured separately and may require overlap or buffering.

A larger fused graph is not inherently faster. The rejected fused-first-frame
graph combined prefill, semantic selection and all residual predictions, passed
the ANE placement gate and produced exact codes, but measured 68.46/81.34 ms
p50/p95 before decoding—slower than separate graphs. Fusion changes scheduling,
memory traffic and compiler choices; benchmark it rather than assuming.

## 7. Build a quality ladder, not one similarity score

For generative audio, use progressively stronger gates:

1. finite tensors and valid shapes;
2. intermediate numerical error against the source implementation;
3. first differing discrete audio-code frame;
4. complete code-sequence equality;
5. raw PCM hash equality;
6. repeated-request determinism;
7. cancellation and state-restoration tests;
8. a text matrix covering short replies, long clauses, technical vocabulary,
   numbers, currency, acronyms and punctuation;
9. listening checks for timbre, age, prosody, pronunciation and artifacts.

Bit-exact PCM against an accepted Core ML baseline is a strong regression gate,
but it does not prove bit identity with upstream PyTorch. Keep these claims
separate in release documentation.

## 8. Preserve streaming semantics

Measure the time when the first real PCM bytes become available, not completion
of a WAV file and not transport headers. One 1,920-sample chunk at 24 kHz is
80 ms of audio in this runtime.

Test at least:

- first PCM and first-to-second PCM intervals;
- generation rate versus playback rate;
- natural EOS versus frame-limit truncation;
- slow or cancelled consumers;
- repeated requests after cancellation;
- one-request-at-a-time enforcement when model state is shared;
- transport timing separately from in-process timing.

Never count prerecorded filler or cached audio as model first-byte latency. If a
deployment uses such a UX technique, report it as a separate end-user metric.

## 9. Package models as immutable artifacts

The Hugging Face release is a runnable artifact, not a dump of the development
directory. It contains only runtime code, declared dependencies, model graphs,
frontend assets, checksums, a manifest and a sample.

Large Core ML packages can share identical weight blobs. This project omits two
byte-identical 887 MB copies and materializes them as hard links in a writable
cache on first load, with a copy fallback across filesystems. Never mutate a
downloaded source snapshot to create those links.

Release safeguards:

- pin the upstream model and source revisions;
- generate SHA-256 checksums and a file manifest;
- verify every configured component exists;
- reject duplicate shared weights;
- reject `__pycache__`, `.pyc` and other test residue;
- keep gRPC/HTTP adapters out of the model artifact;
- run the public API from the assembled folder before publication;
- regenerate the reference prompt and compare raw PCM bytes.

## 10. Failures worth remembering

- A compute-plan pass does not guarantee a speedup.
- Smaller cache capacity may save only a millisecond and add migration cost.
- Resetting a rolling decoder window loses causal convolution/attention history.
- Direct embedding gathers can fall back to CPU even when surrounding matrix
  operations prefer ANE.
- Activation quantization needs representative calibration and can move a graph
  off the desired backend.
- A faster Q8 model can still change speaker character.
- A 4-bit candidate can preserve average numerical similarity while changing
  the first discrete argmax and therefore the whole autoregressive trajectory.
- CPU-only comparison must fail closed when the native runtime crashes; do not
  invent a speedup from an incomplete baseline.
- Long prompts require an explicit capacity and verified state migration; first
  byte and maximum supported duration are separate properties.

## 11. Repeatable ANE-port workflow

For every new model or optimization:

1. pin source and weight revisions;
2. generate source fixtures before exporting Core ML;
3. export one component with explicit shapes and precision;
4. compare source and Core ML outputs;
5. run the strict ANE compute-plan gate;
6. capture real activation samples when calibration is needed;
7. benchmark the isolated component warm and cold;
8. integrate it behind a selectable path;
9. compare full generated codes, PCM, EOS and state restoration;
10. run the stability text matrix and listening gate;
11. assemble a clean release folder and run its public API;
12. publish only claims reproduced by retained reports.

The exact commands and experiment-specific evidence are retained in
[`RESEARCH_LOG.md`](RESEARCH_LOG.md). Structured release measurements live in
[`../BENCHMARKS.json`](../BENCHMARKS.json).

# PrismAlignGPU 0.1.0

## Algorithm

PrismAlignGPU is a deterministic semiglobal progressive protein aligner with a sparse consistency score and iterative profile refinement.

1. FASTA input is retained verbatim apart from inserted output gaps. An optional unknown residue `X` has a distinct code from a gap.
2. Every pair in the small/medium family case receives a full three-state affine Needleman-Wunsch score (gap open 14, extension 0.6, BLOSUM62). Distances are `1 - score / min(self_score_i, self_score_j)`. UPGMA constructs the progressive guide tree. Large CPU-only fallbacks use normalized 3-mer cosine distances.
3. Ordered pairs are independently aligned on the CPU with three affine parameter settings: `(8, 0.6)`, `(12, 0.2)`, and `(14, 0.2)`. Matched residue pairs become sparse direct estimates, averaging model support. A consistency transform adds paths through other sequences. With direct estimate `D`, the stored score is `0.15*D(i,j) + 1.2/N * sum_k D(i,k)D(k,j)`. Entries at most 0.06 are discarded and at most 500 entries are retained per residue. This full transform is used for up to 64 sequences; larger families use at most the 16 most parameter-stable intermediates per residue to bound cubic work. OpenMP parallelizes independent source-sequence rows using only the requested `threads` count.
4. Progressive profile-profile alignment uses column-averaged BLOSUM62 sum-of-pairs scores plus one times the sparse consistency score. Match/insert/delete states use affine penalties; progressive defaults are gap open 12 and extension 0.6. Existing gap frequency lowers local gap penalties. Leading and trailing gaps are free, preventing terminal extensions from forcing false matches.
5. One deterministic single-sequence realignment sweep and up to four random bipartition realignments (fixed seed derived only from family size and alignment length) refine the MSA. Candidates are accepted only when the full BLOSUM62/consistency/gap objective improves.
6. The output writer preserves the original full names, characters, and residue order. Exception handling writes a rectangular left-justified alignment if an unexpected allocation or numerical failure occurs. Inputs with a sequence longer than 20,000 residues use that integrity-safe fallback immediately.

## CUDA implementation

`gpu.cu` implements the substantive all-pairs affine DP used to build the guide tree. Each unordered pair is assigned to one 128-thread CUDA block. Threads fill match and two affine gap matrices along anti-diagonal wavefronts, with block synchronization between diagonals. A block reduction finds the best free-terminal endpoint. Full DP matrices, not k-mer approximations, are used on CUDA device 0.

Batches are capped at 60 million cells across pairs and therefore about 720 MB of DP matrix VRAM, plus small task and sequence buffers. Oversized individual DP jobs or unavailable CUDA initialization cause a CPU fallback. `CUDA_VISIBLE_DEVICES` is never changed. Host sparse-anchor construction and progressive/refinement profile DPs remain on the CPU. CPU and GPU pairwise guide scores were compared directly on all mini pairs after correcting unknown-residue encoding.

## Build and run

```sh
./align/build.sh
./align/run.sh input.fasta output.afa 8
```

`build.sh` is independent of its caller's working directory. With `nvcc`, it compiles for `sm_120` and links CUDA runtime. Without `nvcc`, it builds the CPU-only fallback. No downloaded code, external alignment software, development alignment, or reference alignment is used by the tool.

## Development measurements

Commands used for the final measured run were:

```sh
./align/build.sh
rm -rf dev-output && mkdir dev-output
for d in ../dev/mini/*; do
  f=${d##*/}; mkdir "dev-output/$f"
  /usr/bin/time -f "$f %e %M" ./align/run.sh "$d/in.fasta" "dev-output/$f/out.afa" 8
done
python - <<'PY'
import runpy
from pathlib import Path
score_run = runpy.run_path("../score.py")["score_run"]
dev = Path("../dev/mini")
print(score_run(dev, Path("dev-output"), dev, None))
PY
```

Measured family TC/SP: BB11001 0.9123/0.9415, BB11002 0.0000/0.5987, BB11003 0.5061/0.7213, BB11004 0.4511/0.6431, BB11005 0.1477/0.4813, BB11006 0.3958/0.5067, BB11007 0.6127/0.8261, BB11008 0.6939/0.7381. Collection mean TC was **0.464944** and mean SP **0.682107** over 8 families and 832 core columns. All outputs passed the official integrity checker. Individual wall times were 0.44, 0.43, 0.65, 0.58, 1.67, 0.57, 1.02, and 0.56 seconds (5.92 s total). Highest reported per-process peak RSS was 226,120 KB, including CUDA runtime allocations. These are development-machine measurements, not hidden-data claims.

A synthetic integrity/runtime stress test concatenated five distinct-name copies of BB11005's 14 sequences (70 sequences): 22.06 s and 376,628 KB peak RSS with the final bounded transform (the direct-only version took 18.06 s and 330,900 KB). This was a scaling test only and is not an accuracy measurement. A further 142-sequence stress input completed in 112.56 s with 741,736 KB peak RSS; all 142 names, ungapped residues, and row lengths passed an independent round-trip/integrity check. Two executions of BB11005 were byte-identical, and its forced-CPU output was byte-identical to the normal GPU output. Official score inspection guided the documented gap and consistency tradeoffs: the initial k-mer/progressive baseline was TC 0.287/SP 0.501; pairwise-score guides and the sparse ensemble/transitive score were the main accuracy improvements.

Pair-HMM forward/backward posterior mixtures, neighbor joining, MSA-based tree rebuilding, and deeper random refinement were tested but did not improve the development result and are disabled by default. Single-sequence refinement improved TC from 0.4302 to 0.4390 at the previous consistency setting; the final direct/transitive weighting further improved both mean metrics.

## Limits

This small development set cannot establish hidden-family superiority, especially for BB11002 and BB11005 where exact-column TC remains weak. Full quadratic DPs can be costly on long sequence outliers; the deterministic bounded large-family consistency approximation and explicit oversize fallback prioritize bounded resources and valid outputs. GPU guide scoring reduces pairwise work but does not eliminate the CPU ensemble or progressive cost.

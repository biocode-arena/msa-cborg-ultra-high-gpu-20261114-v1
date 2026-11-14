# PrismAlignGPU

GPU multiple sequence alignment software generated through the biocode-arena harness.

**Unbenchmarked candidate at release.** This source was released before core benchmarking, without quality, speed or ranking claims. Benchmark results may appear later on the biocode-arena leaderboard.

Category: `msa`. Builder: `local/cborg-coder-ultra-high`. Reasoning effort: `high`.

Source tree SHA-256: `e7bc3b74730b13b108376b12daf9ff2b3d2bf8fa3e42b34d1609fdc18c24d4e8`. The published tool is identical to the original development snapshot.

## Use

### Run it on your own data

You need the protein sequences of one family as a FASTA file.
Work in the directory that holds it, and clone this repository into it.
The steps below prepare the input file that the biocode-arena benchmark gave the tool, and run the tool with the command of the benchmark.

The benchmark built and ran the tool on x86-64 Linux with gcc and g++, cmake, make, ninja, Rust, and Python 3.12 with numpy and scipy.
The tool needs no other compiler or library.
If the description above says that the tool runs on a GPU, it also needs an NVIDIA GPU and the CUDA 12.9 toolkit.

The tool reads one file:

| File | Content |
| --- | --- |
| `in.fasta` | The unaligned protein sequences of one family as FASTA text, not compressed. Each header line holds only the sequence name, and each name occurs once. |

The benchmark families held 4 to 142 sequences; larger families are untested.
If the headers of your sequences carry a description after the name, remove it:

```bash
sed '/^>/ s/[[:space:]].*//' family.fasta > in.fasta
```

Build the tool once:

```bash
(cd msa-cborg-ultra-high-gpu-20261114-v1/tool && bash build.sh)
```

Run the tool with three arguments: the input file, the output file and the number of threads.
Give both files with their full paths, as the benchmark did, because a tool may change its directory before it opens them.

```bash
bash msa-cborg-ultra-high-gpu-20261114-v1/tool/run.sh "$PWD/in.fasta" "$PWD/out.afa" 8
```

`out.afa` is the alignment as FASTA text: each input sequence under its name, with `-` for gaps, and all rows of the same length.
The order of the sequences can differ from the input, and the tool may change the case of the letters.
The tool aligns one family in one run; for several families, run it once for each file.

## Status

The README usage check uses development data only.

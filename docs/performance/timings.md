# Alignment throughput

GPU: NVIDIA L40S. CPU: one AMD EPYC 9474F core with Numba/SVML.
Each rate uses one timed run after one untimed warm-up. Molecular preparation,
index construction, and pairwise batch assembly are outside these timers.
GPU runs replay the archived Triton launch configurations. The 100-million-conformer `vol` screening and GPU scaling
measurements additionally pre-read the store before warm-up; that initial read
is excluded from their times.

## Main modes

Alignments per second. CPU screening uses 1,000 conformers, GPU screening ten
million conformers, and GPU pairwise alignment 100,000 pairs. Pairwise results
use 250 query compounds and 400 disjoint library compounds, one conformer each.

| Mode | CPU screen | GPU screen | GPU pairwise |
|---|---:|---:|---:|
| `vol` | 2,470.5 | 1,034,608 | 118,293 |
| `vol_esp` | 841.6 | 368,212 | 82,823 |
| `surf` | 25.7 | 50,726 | 40,090 |
| `surf_esp` | 28.2 | 29,838 | 25,863 |
| `vol_and_surf_esp` | 19.0 | 17,839 | 13,401 |
| `pharm` | 574.1 | 170,039 | 63,658 |
| `vol_color` | 813.0 | 348,793 | 60,728 |
| `vol_lipo` | 468.4 | 209,397 | 58,493 |

## Additional modes evaluated in the SI

CPU screening uses 1,000 conformers; GPU screening and pairwise alignment each
use 100,000 comparisons.

| Mode | CPU screen | GPU screen | GPU pairwise |
|---|---:|---:|---:|
| `vol_pharm` | 280.0 | 98,648 | 39,251 |
| `vol_atomtype` | 267.7 | 258,022 | 57,971 |
| `vol_mr` | 323.2 | 214,050 | 57,154 |
| `surf_tversky` | 20.6 | 38,053 | 31,136 |
| `surf_esp_tversky` | 19.6 | 32,472 | 25,846 |
| `vol_and_surf_esp_tversky` | 19.0 | 18,112 | 13,071 |
| `vol_color_tversky` | 789.8 | 331,031 | 61,817 |
| `vol_lipo_tversky` | 475.3 | 211,608 | 51,847 |
| `pharm_tversky` | 572.6 | 163,818 | 62,641 |
| `vol_fukui` | 461.7 | 213,585 | 58,101 |

`vol_tversky`, `vol_esp_tversky` and `vol_avoid` were not timed. The `_tversky`
names are the registry modes that `similarity="tversky"` runs on the batched backends.

## Parallel screening

CPU scaling uses 10,000 conformers and includes pair construction inside each
persistent worker's timed call. One and 64 workers give 1,881 and 102,959
alignments/s, respectively (54.72-fold speedup). This is the in-memory
`screen_parallel` path, not store-based `screen`.

GPU scaling uses 100 million conformers with one host compute thread per GPU
worker. Times on one, two, and four L40S GPUs are 96.69, 48.49, and 24.55 seconds,
respectively (1.99-fold and 3.94-fold speedups). Each value is one timed run after
warm-up, with worker startup excluded.

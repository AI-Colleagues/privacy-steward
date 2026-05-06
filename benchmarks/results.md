## Throughput Benchmark

Both tools use the same `openai/privacy-filter` model weights.
Benchmarks run on Apple M-series CPU (single process, no GPU).

| Corpus | Size | Tokens | privacy-steward (s) | privacy-steward (tok/s) | opf (s) | opf (tok/s) | Speedup |
|--------|------|--------|--------------------:|------------------------:|--------:|------------:|---------|
| batch_01.txt | 2 KB | 570 | 17.17 | 33 | 46.81 | 12 | 2.73× |
| batch_02.txt | 2 KB | 566 | 22.44 | 25 | 47.94 | 11 | 2.14× |
| batch_03.txt | 2 KB | 562 | 20.41 | 27 | 61.65 | 9 | 3.02× |
| batch_04.txt | 2 KB | 568 | 31.36 | 18 | 52.40 | 10 | 1.67× |
| batch_05.txt | 2 KB | 578 | 28.07 | 20 | 51.39 | 11 | 1.83× |
| batch_06.txt | 2 KB | 562 | 25.95 | 21 | 53.70 | 10 | 2.07× |
| batch_07.txt | 2 KB | 564 | 27.45 | 20 | 59.56 | 9 | 2.17× |
| batch_08.txt | 2 KB | 572 | 33.32 | 17 | 42.69 | 13 | 1.28× |
| batch_09.txt | 2 KB | 557 | 20.81 | 26 | 48.25 | 11 | 2.32× |
| batch_10.txt | 2 KB | 566 | 21.86 | 25 | 44.61 | 12 | 2.04× |

_Speedup = opf_elapsed / privacy-steward_elapsed (higher is better for privacy-steward)._

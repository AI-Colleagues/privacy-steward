## Throughput Benchmark

Both tools use the same `openai/privacy-filter` model weights.
Benchmarks run on Apple M-series CPU (single process, no GPU).

| Corpus | Size | Tokens | privacy-steward (s) | privacy-steward (tok/s) | opf (s) | opf (tok/s) | Speedup |
|--------|------|--------|--------------------:|------------------------:|--------:|------------:|---------|
| 01_emails.txt | 5 KB | 990 | 22.48 | 44 | 37.34 | 26 | 1.66× |
| 02_chat_logs.txt | 6 KB | 1,168 | 33.13 | 35 | 42.16 | 27 | 1.27× |
| 03_support_tickets.txt | 4 KB | 657 | 22.13 | 29 | 29.55 | 22 | 1.34× |
| 04_meeting_notes.txt | 5 KB | 969 | 25.13 | 38 | 33.39 | 29 | 1.33× |
| 05_contracts.txt | 5 KB | 954 | 22.44 | 42 | 29.53 | 32 | 1.32× |
| 06_invoices.txt | 3 KB | 573 | 19.02 | 30 | 27.60 | 20 | 1.45× |
| 07_intake_forms.txt | 4 KB | 750 | 20.84 | 35 | 28.88 | 25 | 1.39× |
| 08_travel_itineraries.txt | 4 KB | 604 | 19.44 | 31 | 27.82 | 21 | 1.43× |
| 09_incident_reports.txt | 5 KB | 836 | 23.25 | 35 | 31.54 | 26 | 1.36× |
| 10_crm_diary.txt | 7 KB | 1,362 | 30.45 | 44 | 40.73 | 33 | 1.34× |

_Speedup = opf_elapsed / privacy-steward_elapsed (higher is better for privacy-steward)._

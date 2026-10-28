## rag

### Retrieval (single-page questions)

| Language | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| tr | 8 | 88% | 100% | 100% | 0.94 |
| en | 9 | 89% | 100% | 100% | 0.94 |
| pt | 9 | 78% | 100% | 100% | 0.87 |
| all | 26 | 85% | 100% | 100% | 0.92 |

### Answers (single-page questions)

| Language | Answerable | Cites a right page | Wrong "not found" | Same language | Unanswerable | "Not found" with sources | "Not found" without sources |
|---|---|---|---|---|---|---|---|
| tr | 8 | 88% | 12% | 100% | 2 | 100% | 0% |
| en | 9 | 89% | 11% | 91% | 2 | 100% | 0% |
| pt | 9 | 89% | 11% | 100% | 1 | 100% | 0% |
| all | 26 | 88% | 12% | 97% | 5 | 100% | 0% |

### Multi-page questions

| Questions | Page coverage | All pages found | Cites all needed pages | Same language |
|---|---|---|---|---|
| 16 | 59% | 25% | 19% | 100% |

Questions that reuse wording from their source page: 0
Answers cut at the token limit: 2

## agent

### Retrieval (single-page questions)

| Language | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| tr | 8 | 88% | 100% | 100% | 0.92 |
| en | 9 | 78% | 100% | 100% | 0.87 |
| pt | 9 | 78% | 89% | 89% | 0.85 |
| all | 26 | 81% | 96% | 96% | 0.88 |

### Answers (single-page questions)

| Language | Answerable | Cites a right page | Wrong "not found" | Same language | Unanswerable | "Not found" with sources |
|---|---|---|---|---|---|---|
| tr | 8 | 88% | 12% | 100% | 2 | 100% |
| en | 9 | 89% | 11% | 100% | 2 | 100% |
| pt | 9 | 78% | 22% | 100% | 1 | 100% |
| all | 26 | 85% | 15% | 100% | 5 | 100% |

### Multi-page questions

| Questions | Page coverage | All pages found | Cites all needed pages | Same language |
|---|---|---|---|---|
| 16 | 95% | 88% | 62% | 100% |

Questions that reuse wording from their source page: 0
Answers cut at the token limit: 0

## multi

### Retrieval (single-page questions)

| Language | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| tr | 8 | 100% | 100% | 100% | 1.00 |
| en | 9 | 100% | 100% | 100% | 1.00 |
| pt | 9 | 78% | 89% | 89% | 0.83 |
| all | 26 | 92% | 96% | 96% | 0.94 |

### Answers (single-page questions)

| Language | Answerable | Cites a right page | Wrong "not found" | Same language | Unanswerable | "Not found" with sources |
|---|---|---|---|---|---|---|
| tr | 8 | 100% | 0% | 100% | 2 | 100% |
| en | 9 | 100% | 0% | 91% | 2 | 100% |
| pt | 9 | 67% | 22% | 100% | 1 | 100% |
| all | 26 | 88% | 8% | 97% | 5 | 100% |

### Multi-page questions

| Questions | Page coverage | All pages found | Cites all needed pages | Same language |
|---|---|---|---|---|
| 16 | 86% | 69% | 12% | 94% |

Questions that reuse wording from their source page: 0
Answers cut at the token limit: 1

## Comparison

| Mode | Single-page Hit@5 | Multi-page coverage | All pages found | Cites all needed pages | "Not found" on unanswerable | Searches | LLM calls | Seconds |
|---|---|---|---|---|---|---|---|---|
| rag | 100% | 59% | 25% | 19% | 100% | 1.0 | 1.0 | 2.7 |
| agent | 100% | 95% | 88% | 62% | 100% | 3.8 | 5.1 | 6.4 |
| multi | 96% | 86% | 69% | 12% | 100% | 2.9 | 2.7 | 4.5 |

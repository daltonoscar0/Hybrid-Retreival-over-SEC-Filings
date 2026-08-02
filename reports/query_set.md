# Query set: 60 candidates cut to 40

`data/queries.jsonl` held 60 candidate queries. RUN.md B2 fixes the set at 40:
"40 keeps the paired tests adequately powered and saves a day of my time". This
records which 20 went and why.

The ten PLAN.md section 2 examples are the seed set and all ten survive
(`q01`-`q10`). The other 30 survivors were chosen to hold the sector mix and the
lexical/semantic mix of the 60 roughly where they were, and to drop
near-duplicates ahead of distinct retrieval behaviours.

**Ids are not renumbered.** A `query_id` is the join key into
`data/pool/pool.jsonl`, both qrel files, and every run file the harness reads.
Renumbering the survivors would silently re-point those joins at the wrong
queries. The surviving ids therefore run `q01`-`q60` with gaps.

## Character classes

Each query's `note` field records what it is meant to exercise. Three classes,
read off those notes:

- **lexical**: the query terms appear near-verbatim in relevant text, so the
  test is term matching and term weighting.
- **semantic**: the concept is phrased many ways with no fixed term, so the test
  is paraphrase matching.
- **mixed**: a fixed term plus something a term match cannot resolve, usually a
  recency qualifier ("in the current period", "newly", "changes to") or a causal
  framing.

## The 20 dropped

Substring counts below are over the 146,449 chunks `data/ticker.duckdb` held
when the selection was run. The Item 7 span repairs in `e2572a7` later took the
corpus to 148,097 chunks. The counts are left as they were, because they are the
record of what the selection actually saw; they are not current corpus counts,
and anything quoting them should say so. A
count is a proxy for how much candidate material exists, not for how much of it
is relevant, and it is quoted only where the material is thin enough that the
query would not fill a judging pool.

| id | query | sector | character | why dropped |
|---|---|---|---|---|
| q16 | automotive semiconductor demand trends | semiconductors | semantic | End-market demand trend, the same retrieval behaviour as q17, which tests it with higher lexical specificity. |
| q20 | advanced packaging and process node transition costs | semiconductors | lexical | Manufacturing-technology jargon already covered by q11; "advanced packaging" hits 172 chunks, the thinnest candidate set in the semiconductor group. |
| q22 | semiconductor equipment lead times | semiconductors | lexical | Near-duplicate of q09 (supply chain disruption affecting component availability), which is a PLAN seed and cannot be dropped. |
| q24 | licensing and royalty revenue from patents | semiconductors | lexical | A single revenue line with one fixed vocabulary; exercises the same exact-term behaviour as q19 and q23 and adds no new failure mode. |
| q25 | distributor channel inventory levels | semiconductors | lexical | Near-duplicate of q12 (inventory write-downs); channel versus company-held inventory is a judging nuance, not a distinct retrieval behaviour. |
| q50 | environmental remediation liabilities | semiconductors | lexical | Environmental disclosure is covered by q43, which is the cross-sector version of the same probe. |
| q35 | branch network reduction and digital banking investment | regional_banks | semantic | Two topics in one query make graded relevance ambiguous: a chunk about branch closures alone is neither clearly 2 nor clearly 3. q34 keeps the strategy-shift behaviour. |
| q36 | wealth management and fee income growth | regional_banks | lexical | A noninterest-income revenue line. q26 already probes bank revenue and this adds no new matching behaviour. |
| q37 | loan loss provision increase | regional_banks | lexical | The income-statement mirror of q27 and q33. Three overlapping credit-quality queries collapse to two. |
| q39 | deposit insurance assessment costs | regional_banks | lexical | Tied to the same 2023-2024 stress episode as q29 and q30, both of which survive, and the narrowest of the three. |
| q40 | credit card and consumer loan delinquency rates | regional_banks | lexical | Consumer credit is a small share of this ten-bank universe's loan books, and the credit-quality behaviour is held by q27 and q33. |
| q41 | executive compensation changes | both | semantic | Compensation detail lives in the proxy statement, which is not in the corpus. "executive compensation" hits 199 chunks, nearly all risk-factor boilerplate. |
| q42 | board of directors changes | both | lexical | The change events are disclosed in 8-K Item 5.02, which is not among the extracted sections. The 3,476 chunks matching "board of directors" are undifferentiated governance boilerplate with no clean relevance criterion. |
| q46 | stock-based compensation expense trends | both | lexical | Expense detail lives in the notes to the financial statements, not the item sections; the MD&A mentions are formulaic. |
| q48 | pension plan funded status | both | lexical | "funded status" hits 54 chunks. Too thin to fill a judging pool. |
| q49 | related party transactions | both | lexical | "related part" hits 42 chunks, and the disclosure itself lives in the proxy. |
| q51 | intellectual property litigation | both | semantic | Near-duplicate of q04 (newly disclosed litigation, a PLAN seed) crossed with q52 (antitrust investigation). Both survive. |
| q53 | credit rating downgrade risk | both | semantic | Subsumed by q54, which covers the same financing-risk narrative over a wider set of phrasings. |
| q55 | foreign currency hedging program | both | lexical | Near-duplicate of q05 (foreign exchange impact on reported revenue), a PLAN seed. |
| q58 | going concern doubt | both | lexical | "going concern" hits 16 of 146,449 chunks. A query with almost no relevant material gives a degenerate per-query nDCG. q59 (material weakness in internal controls, 284 chunks) keeps the low-base-rate precision test. |

Dropped by sector: 6 semiconductors, 5 regional_banks, 9 both.
Dropped by character: 12 lexical, 8 semantic, 0 mixed. Every mixed query is
either a PLAN seed or the only recency-qualifier probe in its sector.

## The surviving 40

| sector | lexical | semantic | mixed | total |
|---|---|---|---|---|
| both | 9 | 5 | 5 | 19 |
| semiconductors | 7 | 3 | 1 | 11 |
| regional_banks | 6 | 3 | 1 | 10 |
| **total** | **22** | **11** | **7** | **40** |

Shares before and after the cut:

| split | 60-query set | 40-query set |
|---|---|---|
| both | 46.7% | 47.5% |
| semiconductors | 28.3% | 27.5% |
| regional_banks | 25.0% | 25.0% |
| lexical | 61.7% | 55.0% |
| semantic | 26.7% | 27.5% |
| mixed | 11.7% | 17.5% |

The sector mix is unchanged to within a percentage point. The character mix
moves: mixed rises 5.8 points and lexical falls 6.7. That is forced by the seed
set. Five of the ten PLAN examples are mixed and none of them can be dropped, so
they are 12.5% of the 40 on their own.

Full survivor list, by sector.

**both (19)**

| id | query | character |
|---|---|---|
| q01 | customer concentration risk | lexical |
| q02 | goodwill impairment charge in the current period | mixed |
| q03 | management commentary on gross margin compression | semantic |
| q04 | newly disclosed litigation | mixed |
| q05 | foreign exchange impact on reported revenue | mixed |
| q06 | changes to revenue recognition policy | semantic |
| q07 | data center capital expenditure plans | mixed |
| q08 | cybersecurity incident disclosure | lexical |
| q10 | share repurchase authorization increase | lexical |
| q43 | climate change risk disclosure | lexical |
| q44 | data privacy regulation compliance | semantic |
| q45 | workforce reduction and restructuring charges | lexical |
| q47 | effective tax rate changes | mixed |
| q52 | antitrust investigation | lexical |
| q54 | liquidity and access to capital markets | semantic |
| q56 | segment reporting changes | lexical |
| q57 | impairment of long-lived assets | lexical |
| q59 | material weakness in internal controls | lexical |
| q60 | why did operating expenses increase this quarter | semantic |

**semiconductors (11)**

| id | query | character |
|---|---|---|
| q09 | supply chain disruption affecting component availability | mixed |
| q11 | wafer fabrication capacity constraints | lexical |
| q12 | inventory write-downs due to excess supply | semantic |
| q13 | export restrictions on sales to China | lexical |
| q14 | foundry versus integrated device manufacturing strategy | semantic |
| q15 | research and development spending as a share of revenue | lexical |
| q17 | data center GPU demand growth | lexical |
| q18 | pricing pressure from competing chip suppliers | semantic |
| q19 | order backlog and cancellations | lexical |
| q21 | CHIPS Act government incentives | lexical |
| q23 | design wins with major customers | lexical |

**regional_banks (10)**

| id | query | character |
|---|---|---|
| q26 | net interest margin compression | lexical |
| q27 | allowance for credit losses methodology change | mixed |
| q28 | commercial real estate loan concentration | lexical |
| q29 | deposit outflows and funding cost pressure | semantic |
| q30 | unrealized losses on available-for-sale securities | lexical |
| q31 | interest rate risk management strategy | semantic |
| q32 | capital ratios and stress test results | lexical |
| q33 | nonperforming loan trends | lexical |
| q34 | bank merger and acquisition activity | semantic |
| q38 | Federal Reserve discount window borrowing | lexical |

## Status

These 40 are still candidates, in the same sense the 60 were: PLAN.md says the
query set is hand-verified. Edit the text, the `sector`, or the `note` before
pooling for real. Adding a query back is cheap. Adding one after judging has
started is not, because it needs its own pool and its own judging pass.

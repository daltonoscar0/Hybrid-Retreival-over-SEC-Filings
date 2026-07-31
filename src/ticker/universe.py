"""The locked 20-company, 2-sector corpus universe.

PLAN.md section 1 fixes this at 20 companies, 6 years, 2 sectors and says stop
revisiting it. Two sectors: semiconductors and regional banks, both with rich
risk-factor language and genuine period-over-period change, and both give a
natural sector-matched background for the Phase 4 novelty contrast.

Picked in 2026 from companies still listed and filing continuously under one
CIK from 2020 through 2025. This is a survivorship-biased sample by
construction: any company in either sector that was acquired, delisted, or
failed during the window is excluded, because it cannot have "continuous
filing history" through the end of the window. Concretely, this ruled out
Marvell (redomiciled in 2021, orphaning its pre-2022 CIK), Xilinx and Maxim
Integrated (acquired), and every regional bank that failed or was acquired in
the 2023 stress episode (SVB Financial, Signature Bank, First Republic) or
merged (People's United, South State/CenterState). That limitation is
documented here, not fixed: fixing it would mean deliberately including
companies whose filings stopped, which breaks the "6 years of filings per
firm" assumption the per-firm language models in Phase 4 depend on.

Every CIK below was checked against EDGAR before being locked in: each filed
exactly 6 10-Ks and 18 10-Qs with filing dates in [2020-01-01, 2025-12-31]
under this CIK, amendments excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

SEMICONDUCTORS = "semiconductors"
REGIONAL_BANKS = "regional_banks"

SECTORS = (SEMICONDUCTORS, REGIONAL_BANKS)

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"


@dataclass(frozen=True, slots=True)
class Company:
    ticker: str
    cik: int
    sector: str


UNIVERSE: tuple[Company, ...] = (
    # semiconductors
    Company("INTC", 50863, SEMICONDUCTORS),
    Company("NVDA", 1045810, SEMICONDUCTORS),
    Company("TXN", 97476, SEMICONDUCTORS),
    Company("QCOM", 804328, SEMICONDUCTORS),
    Company("AVGO", 1730168, SEMICONDUCTORS),
    Company("MU", 723125, SEMICONDUCTORS),
    Company("ADI", 6281, SEMICONDUCTORS),
    Company("NXPI", 1413447, SEMICONDUCTORS),
    Company("MCHP", 827054, SEMICONDUCTORS),
    Company("ON", 1097864, SEMICONDUCTORS),
    # regional banks
    Company("ZION", 109380, REGIONAL_BANKS),
    Company("RF", 1281761, REGIONAL_BANKS),
    Company("HBAN", 49196, REGIONAL_BANKS),
    Company("KEY", 91576, REGIONAL_BANKS),
    Company("FITB", 35527, REGIONAL_BANKS),
    Company("CFG", 759944, REGIONAL_BANKS),
    Company("WAL", 1212545, REGIONAL_BANKS),
    Company("EWBC", 1069157, REGIONAL_BANKS),
    Company("CFR", 39263, REGIONAL_BANKS),
    Company("CMA", 28412, REGIONAL_BANKS),
)

assert len(UNIVERSE) == 20
assert sum(1 for c in UNIVERSE if c.sector == SEMICONDUCTORS) == 10
assert sum(1 for c in UNIVERSE if c.sector == REGIONAL_BANKS) == 10
assert len({c.cik for c in UNIVERSE}) == 20


def by_cik(cik: int) -> Company | None:
    for company in UNIVERSE:
        if company.cik == cik:
            return company
    return None

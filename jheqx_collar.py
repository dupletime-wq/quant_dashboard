from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
import gzip
from io import BytesIO, StringIO
import json
import os
import re
import statistics
import tempfile
from typing import Any
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zlib

try:
    import streamlit as st
except Exception:
    st = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


SEC_BASE_URL = "https://www.sec.gov"
SEC_SERIES_ID = "S000043249"
JPM_MONTHLY_HOLDINGS_URL = (
    "https://am.jpmorgan.com/FundsMarketingHandler/pdf?"
    "country=us&cusip=46637K281&locale=en-US&role=adv&type=monthlyMFHoldings"
)
NEOS_QQQI_PAGE_URL = "https://neosfunds.com/qqqi/"
NEOS_QQQI_HOLDINGS_CSV_URL = (
    "https://neosfunds.com/wp-admin/admin-ajax.php?action=download_holdings_csv&ticker=qqqi"
)
JEPQ_FACT_SHEET_URL = (
    "https://am.jpmorgan.com/content/dam/jpm-am-aem/americas/us/en/literature/fact-sheet/etfs/FS-JEPQ.PDF"
)
JEPQ_DAILY_HOLDINGS_URL = (
    "https://am.jpmorgan.com/FundsMarketingHandler/pdf?"
    "country=us&cusip=46654Q203&locale=en-US&role=adv&type=dailyETFHoldings"
)
SEC_HEADERS = {
    "User-Agent": "ProjectResearch/1.0 contact example@example.com",
    "Host": "www.sec.gov",
}
NEOS_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml,text/csv;q=0.9,*/*;q=0.8",
}
JPM_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/pdf",
}
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}
CACHE_TTL_SECONDS = 60 * 60 * 24
CACHE_DIR = os.path.join(tempfile.gettempdir(), "jheqx_collar_cache")
NS = {"nport": "http://www.sec.gov/edgar/nport"}
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_VISIBLE_QUARTERS = 8


@dataclass
class QuarterlyCollarSnapshot:
    quarter_label: str
    hedge_start: date
    hedge_end: date
    as_of_date: date
    spx_level: float | None
    long_put: float | None
    short_put: float | None
    short_call: float | None
    source_url: str
    source_kind: str
    is_disclosed: bool
    is_estimated: bool


@dataclass
class OverlayLeg:
    series: str
    strike: float
    expiry_date: date | None = None
    contracts: int | None = None
    source_label: str | None = None


@dataclass
class OverlaySnapshot:
    underlying_symbol: str
    quarter_label: str
    hedge_start: date
    hedge_end: date
    as_of_date: date
    underlying_level: float | None
    long_put: float | None
    short_put: float | None
    short_call: float | None
    source_url: str
    source_kind: str
    is_disclosed: bool
    is_estimated: bool
    legs: tuple[OverlayLeg, ...] = field(default_factory=tuple)


@dataclass
class FlowReference:
    ticker: str
    name: str
    aum: float | None
    as_of_date: date | None
    strategy_note: str
    visibility: str
    source_url: str


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _read_json_cache(name: str) -> Any | None:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_json_cache(name: str, payload: Any) -> None:
    path = _cache_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _request_text(url: str, headers: dict[str, str]) -> str:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = response.read()
        content_encoding = (response.headers.get("Content-Encoding") or "").lower()
        if content_encoding == "gzip":
            payload = gzip.decompress(payload)
        elif content_encoding == "deflate":
            try:
                payload = zlib.decompress(payload)
            except zlib.error:
                payload = zlib.decompress(payload, -zlib.MAX_WBITS)
        return payload.decode("utf-8", errors="replace")


def _request_bytes(url: str, headers: dict[str, str]) -> bytes:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = response.read()
        content_encoding = (response.headers.get("Content-Encoding") or "").lower()
        if content_encoding == "gzip":
            payload = gzip.decompress(payload)
        elif content_encoding == "deflate":
            try:
                payload = zlib.decompress(payload)
            except zlib.error:
                payload = zlib.decompress(payload, -zlib.MAX_WBITS)
        return payload


def _request_json(url: str, headers: dict[str, str]) -> Any:
    return json.loads(_request_text(url, headers))


def _parse_us_date(value: str) -> date:
    return datetime.strptime(value, "%m/%d/%Y").date()


def _quarter_label(value: date) -> str:
    return f"{value.year} Q{((value.month - 1) // 3) + 1}"


def _quarter_start(value: date) -> date:
    month = ((value.month - 1) // 3) * 3 + 1
    return date(value.year, month, 1)


def _quarter_end(value: date) -> date:
    start = _quarter_start(value)
    next_month = start.month + 3
    next_year = start.year
    if next_month > 12:
        next_month -= 12
        next_year += 1
    return date(next_year, next_month, 1) - timedelta(days=1)


def _next_quarter_start(value: date) -> date:
    start = _quarter_start(value)
    next_month = start.month + 3
    next_year = start.year
    if next_month > 12:
        next_month -= 12
        next_year += 1
    return date(next_year, next_month, 1)


def _first_market_day(price_rows: list[dict[str, Any]], start: date, end: date) -> date:
    for row in price_rows:
        row_date = row["date"]
        if start <= row_date <= end:
            return row_date
    return start


def _find_close_on_or_before(price_rows: list[dict[str, Any]], target: date) -> float | None:
    candidate: float | None = None
    for row in price_rows:
        row_date = row["date"]
        if row_date > target:
            break
        candidate = float(row["close"])
    return candidate


def _serialize_snapshot(snapshot: QuarterlyCollarSnapshot) -> dict[str, Any]:
    payload = asdict(snapshot)
    payload["hedge_start"] = snapshot.hedge_start.isoformat()
    payload["hedge_end"] = snapshot.hedge_end.isoformat()
    payload["as_of_date"] = snapshot.as_of_date.isoformat()
    return payload


def _deserialize_snapshot(payload: dict[str, Any]) -> QuarterlyCollarSnapshot:
    return QuarterlyCollarSnapshot(
        quarter_label=str(payload["quarter_label"]),
        hedge_start=date.fromisoformat(payload["hedge_start"]),
        hedge_end=date.fromisoformat(payload["hedge_end"]),
        as_of_date=date.fromisoformat(payload["as_of_date"]),
        spx_level=float(payload["spx_level"]) if payload.get("spx_level") is not None else None,
        long_put=float(payload["long_put"]) if payload.get("long_put") is not None else None,
        short_put=float(payload["short_put"]) if payload.get("short_put") is not None else None,
        short_call=float(payload["short_call"]) if payload.get("short_call") is not None else None,
        source_url=str(payload["source_url"]),
        source_kind=str(payload["source_kind"]),
        is_disclosed=bool(payload["is_disclosed"]),
        is_estimated=bool(payload["is_estimated"]),
    )


def _serialize_overlay_snapshot(snapshot: OverlaySnapshot) -> dict[str, Any]:
    payload = asdict(snapshot)
    payload["hedge_start"] = snapshot.hedge_start.isoformat()
    payload["hedge_end"] = snapshot.hedge_end.isoformat()
    payload["as_of_date"] = snapshot.as_of_date.isoformat()
    payload["legs"] = [
        {
            "series": leg.series,
            "strike": leg.strike,
            "expiry_date": leg.expiry_date.isoformat() if leg.expiry_date else None,
            "contracts": leg.contracts,
            "source_label": leg.source_label,
        }
        for leg in snapshot.legs
    ]
    return payload


def _deserialize_overlay_snapshot(payload: dict[str, Any]) -> OverlaySnapshot:
    return OverlaySnapshot(
        underlying_symbol=str(payload["underlying_symbol"]),
        quarter_label=str(payload["quarter_label"]),
        hedge_start=date.fromisoformat(payload["hedge_start"]),
        hedge_end=date.fromisoformat(payload["hedge_end"]),
        as_of_date=date.fromisoformat(payload["as_of_date"]),
        underlying_level=float(payload["underlying_level"]) if payload.get("underlying_level") is not None else None,
        long_put=float(payload["long_put"]) if payload.get("long_put") is not None else None,
        short_put=float(payload["short_put"]) if payload.get("short_put") is not None else None,
        short_call=float(payload["short_call"]) if payload.get("short_call") is not None else None,
        source_url=str(payload["source_url"]),
        source_kind=str(payload["source_kind"]),
        is_disclosed=bool(payload["is_disclosed"]),
        is_estimated=bool(payload["is_estimated"]),
        legs=tuple(
            OverlayLeg(
                series=str(leg["series"]),
                strike=float(leg["strike"]),
                expiry_date=date.fromisoformat(leg["expiry_date"]) if leg.get("expiry_date") else None,
                contracts=int(leg["contracts"]) if leg.get("contracts") is not None else None,
                source_label=str(leg["source_label"]) if leg.get("source_label") else None,
            )
            for leg in payload.get("legs", [])
        ),
    )


def _serialize_flow_reference(reference: FlowReference) -> dict[str, Any]:
    return {
        "ticker": reference.ticker,
        "name": reference.name,
        "aum": reference.aum,
        "as_of_date": reference.as_of_date.isoformat() if reference.as_of_date else None,
        "strategy_note": reference.strategy_note,
        "visibility": reference.visibility,
        "source_url": reference.source_url,
    }


def _deserialize_flow_reference(payload: dict[str, Any]) -> FlowReference:
    return FlowReference(
        ticker=str(payload["ticker"]),
        name=str(payload["name"]),
        aum=float(payload["aum"]) if payload.get("aum") is not None else None,
        as_of_date=date.fromisoformat(payload["as_of_date"]) if payload.get("as_of_date") else None,
        strategy_note=str(payload["strategy_note"]),
        visibility=str(payload["visibility"]),
        source_url=str(payload["source_url"]),
    )


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed")
    reader = PdfReader(BytesIO(pdf_bytes))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts)


def _parse_money_value(value: str) -> float:
    cleaned = value.replace("$", "").replace(",", "").strip()
    if not cleaned:
        raise ValueError("Missing money value")
    return float(cleaned)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.replace("$", "").replace(",", "").replace("%", "").strip()
    if not cleaned:
        return None
    return float(cleaned)


def _extract_qqqi_page_metadata(html: str) -> dict[str, Any]:
    as_of_match = re.search(r"Fund Details As of:\s*(\d{2}/\d{2}/\d{4})", html, re.I)
    net_assets_match = re.search(r"Net Assets\s*\$([\d,]+(?:\.\d+)?)", html, re.I)
    strategy_match = re.search(
        r"Targeted Benefits of QQQI's Strategy:[\s\S]{0,600}?sold and purchased NDX index options",
        html,
        re.I,
    )
    return {
        "as_of_date": _parse_us_date(as_of_match.group(1)) if as_of_match else None,
        "net_assets": _parse_money_value(net_assets_match.group(1)) if net_assets_match else None,
        "strategy_note": (
            "Active NDX options overlay; the public page states the strategy may use both sold and purchased NDX index options."
            if strategy_match
            else "Active NDX options overlay with current holdings disclosed on the issuer site."
        ),
    }


def _parse_occ_option_symbol(symbol: str) -> tuple[str, date, str, float] | None:
    compact = re.sub(r"\s+", "", symbol or "").upper()
    match = re.match(r"^(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$", compact)
    if not match:
        return None
    expiry = datetime.strptime(match.group("expiry"), "%y%m%d").date()
    strike = int(match.group("strike")) / 1000.0
    return match.group("root"), expiry, match.group("cp"), strike


def _primary_leg_strike(legs: list[OverlayLeg], prefix: str) -> float | None:
    matches = [leg.strike for leg in legs if leg.series.startswith(prefix)]
    if not matches:
        return None
    return min(matches)


def _parse_qqqi_snapshot_from_holdings_csv(csv_text: str, metadata: dict[str, Any]) -> OverlaySnapshot:
    reader = csv.DictReader(StringIO(csv_text))
    rows = list(reader)
    if not rows:
        raise ValueError("QQQI holdings CSV returned no rows")

    as_of_date = _parse_us_date(rows[0]["Date"]) if rows[0].get("Date") else metadata.get("as_of_date")
    if as_of_date is None:
        raise ValueError("QQQI holdings CSV did not include an as-of date")

    legs: list[OverlayLeg] = []
    for row in rows:
        occ_symbol = row.get("StockTicker") or ""
        parsed = _parse_occ_option_symbol(occ_symbol)
        if not parsed:
            continue
        root, expiry_date, option_side, strike = parsed
        if root != "NDX":
            continue

        contracts_value = int(float((row.get("Shares") or "0").replace(",", "")))
        if contracts_value == 0:
            continue

        if option_side == "C" and contracts_value < 0:
            series = f"Short Call {strike:,.0f}"
        elif option_side == "C" and contracts_value > 0:
            series = f"Long Call {strike:,.0f}"
        elif option_side == "P" and contracts_value < 0:
            series = f"Short Put {strike:,.0f}"
        else:
            series = f"Long Put {strike:,.0f}"

        legs.append(
            OverlayLeg(
                series=series,
                strike=strike,
                expiry_date=expiry_date,
                contracts=abs(contracts_value),
                source_label=row.get("SecurityName") or None,
            )
        )

    latest_expiry = max((leg.expiry_date for leg in legs if leg.expiry_date is not None), default=as_of_date)
    return OverlaySnapshot(
        underlying_symbol="^NDX",
        quarter_label=_quarter_label(as_of_date),
        hedge_start=as_of_date,
        hedge_end=latest_expiry,
        as_of_date=as_of_date,
        underlying_level=None,
        long_put=_primary_leg_strike(legs, "Long Put"),
        short_put=_primary_leg_strike(legs, "Short Put"),
        short_call=_primary_leg_strike(legs, "Short Call"),
        source_url=NEOS_QQQI_HOLDINGS_CSV_URL,
        source_kind="NEOS Daily Holdings CSV",
        is_disclosed=True,
        is_estimated=False,
        legs=tuple(sorted(legs, key=lambda leg: (leg.series, leg.strike))),
    )


def _estimate_net_assets_from_holdings_text(pdf_text: str) -> float | None:
    estimates: list[float] = []
    for match in re.finditer(
        r"\$\s*(?P<value>[\d,]+\.\d{2})[\s\S]{0,80}?(?P<pct_value>\d+\.\d+)%\s+(?P<pct_assets>\d+\.\d+)%",
        pdf_text,
    ):
        value = _parse_money_value(match.group("value"))
        pct_assets = float(match.group("pct_assets"))
        if pct_assets <= 0:
            continue
        estimates.append(value / (pct_assets / 100.0))
    if not estimates:
        return None
    return statistics.median(estimates[:20])


def _parse_snapshot_from_monthly_holdings(pdf_text: str) -> QuarterlyCollarSnapshot:
    as_of_match = re.search(r"As of Date:\s*(\d{2}/\d{2}/\d{4})", pdf_text)
    if not as_of_match:
        raise ValueError("Could not find the monthly holdings as-of date")
    as_of_date = _parse_us_date(as_of_match.group(1))

    option_pattern = re.compile(
        r"SPXW\s+(?P<option_type>PUT|CALL)\s+USD\s+(?P<expiry>\d{2}/\d{2}/\d{4})\s+"
        r"INDEX\s+(?:PUT|CALL)\s+OPTION\s+Physical\s+(?P<contracts>-?\d[\d,]*)\s+"
        r"(?P<market_value>-?\$?\s*[\d,]+\.\d{2})[\s\S]{0,120}?100\s+\$\s*(?P<strike>[\d,]+\.\d{2})",
        re.I,
    )

    long_put: float | None = None
    short_put: float | None = None
    short_call: float | None = None
    for match in option_pattern.finditer(pdf_text):
        option_type = match.group("option_type").upper()
        contracts = int(match.group("contracts").replace(",", ""))
        strike = float(match.group("strike").replace(",", ""))
        if option_type == "PUT" and contracts > 0:
            long_put = strike
        elif option_type == "PUT" and contracts < 0:
            short_put = strike
        elif option_type == "CALL" and contracts < 0:
            short_call = strike

    if long_put is None or short_put is None or short_call is None:
        raise ValueError("Could not extract the full collar from the monthly holdings PDF")

    return QuarterlyCollarSnapshot(
        quarter_label=_quarter_label(as_of_date),
        hedge_start=_quarter_start(as_of_date),
        hedge_end=_quarter_end(as_of_date),
        as_of_date=as_of_date,
        spx_level=None,
        long_put=long_put,
        short_put=short_put,
        short_call=short_call,
        source_url=JPM_MONTHLY_HOLDINGS_URL,
        source_kind="JPM Monthly Holdings",
        is_disclosed=True,
        is_estimated=False,
    )


def _parse_filing_rows(html: str) -> list[str]:
    pattern = re.compile(
        r'<tr(?: class="evenRow")?>\s*'
        r'<td nowrap="nowrap">NPORT-P</td>\s*'
        r'<td nowrap="nowrap"><a href="(?P<href>/Archives/edgar/data/[^"]+-index\.htm)"[^>]*>.*?</a></td>',
        re.S,
    )
    return [SEC_BASE_URL + match.group("href") for match in pattern.finditer(html)]


def _parse_filing_rows_from_atom(atom_text: str) -> list[str]:
    try:
        root = ET.fromstring(atom_text)
        urls: list[str] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            for link in entry.findall("atom:link", ATOM_NS):
                href = link.attrib.get("href", "")
                rel = link.attrib.get("rel", "")
                if rel == "alternate" and href.endswith("-index.htm"):
                    urls.append(href)
                    break
        if urls:
            return urls
    except ET.ParseError:
        pass

    pattern = re.compile(r"<filing-href>(https://www\.sec\.gov/Archives/edgar/data/[^<]+-index\.htm)</filing-href>")
    return [match.group(1) for match in pattern.finditer(atom_text)]


def _extract_xml_url(index_html: str) -> str | None:
    matches = re.findall(r'href="(?P<href>/Archives/edgar/data/[^"]+/primary_doc\.xml)"', index_html)
    if not matches:
        return None
    preferred = next((href for href in matches if "/xslForm" not in href and "/xsl" not in href), matches[-1])
    return SEC_BASE_URL + preferred


def _extract_report_date_from_index(index_html: str) -> date | None:
    match = re.search(r"Period of Report</div>\s*<div class=\"info\">\s*([^<\s]+)", index_html, re.S)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1).strip())
    except ValueError:
        return None


def _parse_snapshot_from_xml(
    xml_text: str,
    source_url: str,
    report_date_override: date | None = None,
) -> QuarterlyCollarSnapshot:
    root = ET.fromstring(xml_text)
    rep_pd_end = root.findtext(".//nport:genInfo/nport:repPdEnd", namespaces=NS)
    rep_pd_date = root.findtext(".//nport:genInfo/nport:repPdDate", namespaces=NS)
    series_name = root.findtext(".//nport:genInfo/nport:seriesName", namespaces=NS) or ""
    if not rep_pd_date and not rep_pd_end and report_date_override is None:
        raise ValueError("No report date found in SEC XML")
    if "JPMorgan Hedged Equity Fund" not in series_name:
        raise ValueError(f"Unexpected series in filing: {series_name or 'unknown'}")

    as_of_date: date | None = None
    if rep_pd_date:
        as_of_date = date.fromisoformat(rep_pd_date)
    elif report_date_override is not None:
        as_of_date = report_date_override
    elif rep_pd_end:
        as_of_date = date.fromisoformat(rep_pd_end)
    if as_of_date is None:
        raise ValueError("Unable to determine filing as-of date")

    long_put: float | None = None
    short_put: float | None = None
    short_call: float | None = None

    for security in root.findall(".//nport:invstOrSec", namespaces=NS):
        asset_cat = security.findtext("nport:assetCat", namespaces=NS)
        if asset_cat != "DE":
            continue

        title = security.findtext("nport:title", namespaces=NS) or ""
        index_name = security.findtext(".//nport:indexBasketInfo/nport:indexName", namespaces=NS) or ""
        if "S&P 500 Index" not in title and "S&P 500 Index" not in index_name:
            continue

        put_or_call = security.findtext(".//nport:optionSwaptionWarrantDeriv/nport:putOrCall", namespaces=NS)
        written_or_purchased = security.findtext(".//nport:optionSwaptionWarrantDeriv/nport:writtenOrPur", namespaces=NS)
        exercise_price = security.findtext(".//nport:optionSwaptionWarrantDeriv/nport:exercisePrice", namespaces=NS)
        if not put_or_call or not written_or_purchased or not exercise_price:
            continue

        strike = float(exercise_price)
        if written_or_purchased == "Purchased" and put_or_call == "Put":
            long_put = strike
        elif written_or_purchased == "Written" and put_or_call == "Put":
            short_put = strike
        elif written_or_purchased == "Written" and put_or_call == "Call":
            short_call = strike

    if long_put is None or short_put is None or short_call is None:
        raise ValueError("Could not extract full collar strikes from SEC XML")

    return QuarterlyCollarSnapshot(
        quarter_label=_quarter_label(as_of_date),
        hedge_start=_quarter_start(as_of_date),
        hedge_end=_quarter_end(as_of_date),
        as_of_date=as_of_date,
        spx_level=None,
        long_put=long_put,
        short_put=short_put,
        short_call=short_call,
        source_url=source_url,
        source_kind="SEC N-PORT",
        is_disclosed=True,
        is_estimated=False,
    )


if st is not None:

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def load_jheqx_quarterly_snapshots(limit: int = 12) -> list[QuarterlyCollarSnapshot]:
        filings_url = (
            f"{SEC_BASE_URL}/cgi-bin/browse-edgar?"
            f"action=getcompany&CIK={SEC_SERIES_ID}&type=NPORT-P&owner=exclude&count=40"
        )
        try:
            browse_html = _request_text(filings_url, SEC_HEADERS)
            filing_urls = _parse_filing_rows(browse_html)
            if not filing_urls:
                atom_text = _request_text(f"{filings_url}&output=atom", SEC_HEADERS)
                filing_urls = _parse_filing_rows_from_atom(atom_text)

            snapshots: list[QuarterlyCollarSnapshot] = []
            seen_as_of_dates: set[date] = set()
            errors: list[str] = []
            for index_url in filing_urls:
                try:
                    index_html = _request_text(index_url, SEC_HEADERS)
                    index_report_date = _extract_report_date_from_index(index_html)
                    xml_url = _extract_xml_url(index_html)
                    if not xml_url:
                        errors.append(f"{index_url}: primary_doc.xml not found")
                        continue
                    xml_text = _request_text(xml_url, SEC_HEADERS)
                    snapshot = _parse_snapshot_from_xml(
                        xml_text,
                        index_url,
                        report_date_override=index_report_date,
                    )
                except Exception as exc:
                    errors.append(f"{index_url}: {exc}")
                    continue

                if snapshot.as_of_date in seen_as_of_dates:
                    continue
                seen_as_of_dates.add(snapshot.as_of_date)
                snapshots.append(snapshot)
                if len(snapshots) >= limit:
                    break
            if not snapshots:
                detail = errors[0] if errors else "No filing rows matched on the SEC response"
                raise ValueError(f"No JHEQX quarterly snapshots were extracted. {detail}")
            _write_json_cache("snapshots.json", [_serialize_snapshot(item) for item in snapshots])
            return snapshots
        except Exception:
            cached = _read_json_cache("snapshots.json")
            if not cached:
                raise
            return [_deserialize_snapshot(item) for item in cached][:limit]


    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def load_latest_monthly_snapshot() -> QuarterlyCollarSnapshot | None:
        cache_name = "latest_monthly_snapshot.json"
        try:
            pdf_bytes = _request_bytes(JPM_MONTHLY_HOLDINGS_URL, JPM_HEADERS)
            pdf_text = _extract_pdf_text(pdf_bytes)
            snapshot = _parse_snapshot_from_monthly_holdings(pdf_text)
            _write_json_cache(cache_name, _serialize_snapshot(snapshot))
            return snapshot
        except Exception:
            cached = _read_json_cache(cache_name)
            if not cached:
                return None
            return _deserialize_snapshot(cached)


    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def load_market_prices(symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        period_start = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        period_end = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1={period_start}&period2={period_end}&interval=1d&includeAdjustedClose=true"
        )
        cache_name = f"market_{symbol.replace('^', 'idx_')}.json"
        try:
            payload = _request_json(url, YAHOO_HEADERS)
            result = (payload.get("chart") or {}).get("result") or []
            if not result:
                raise ValueError(f"No market data returned for {symbol}")
            series = result[0]
            timestamps = series.get("timestamp") or []
            quote = ((series.get("indicators") or {}).get("quote") or [{}])[0]
            adjclose = ((series.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
            closes = quote.get("close") or []
            rows: list[dict[str, Any]] = []
            for index, timestamp in enumerate(timestamps):
                close_value = adjclose[index] if index < len(adjclose) else None
                if close_value is None and index < len(closes):
                    close_value = closes[index]
                if close_value is None:
                    continue
                row_date = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).date()
                rows.append({"date": row_date, "close": float(close_value)})
            if not rows:
                raise ValueError(f"All market prices were empty for {symbol}")
            _write_json_cache(
                cache_name,
                [{"date": row["date"].isoformat(), "close": row["close"]} for row in rows],
            )
            return rows
        except Exception:
            cached = _read_json_cache(cache_name)
            if not cached:
                raise
            return [{"date": date.fromisoformat(str(row["date"])), "close": float(row["close"])} for row in cached]


    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def load_qqqi_public_snapshot() -> OverlaySnapshot | None:
        cache_name = "qqqi_public_snapshot.json"
        try:
            page_html = _request_text(NEOS_QQQI_PAGE_URL, NEOS_HEADERS)
            metadata = _extract_qqqi_page_metadata(page_html)
            holdings_csv = _request_text(NEOS_QQQI_HOLDINGS_CSV_URL, NEOS_HEADERS)
            snapshot = _parse_qqqi_snapshot_from_holdings_csv(holdings_csv, metadata)
            _write_json_cache(cache_name, _serialize_overlay_snapshot(snapshot))
            return snapshot
        except Exception:
            cached = _read_json_cache(cache_name)
            if not cached:
                return None
            return _deserialize_overlay_snapshot(cached)


    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def load_jepq_flow_reference() -> FlowReference | None:
        cache_name = "jepq_flow_reference.json"
        try:
            holdings_pdf = _request_bytes(JEPQ_DAILY_HOLDINGS_URL, JPM_HEADERS)
            holdings_text = _extract_pdf_text(holdings_pdf)
            as_of_match = re.search(r"As of Date:\s*(\d{2}/\d{2}/\d{4})", holdings_text)
            as_of_date = _parse_us_date(as_of_match.group(1)) if as_of_match else None
            aum = _estimate_net_assets_from_holdings_text(holdings_text)
            eln_count = len(re.findall(r"Equity Linked Notes Synthetic", holdings_text))
            note = (
                f"ELN-based Nasdaq income strategy. The current daily holdings text shows {eln_count} synthetic ELN rows, "
                "so exact current option strike spreads are not directly observable."
            )
            reference = FlowReference(
                ticker="JEPQ",
                name="JPMorgan Nasdaq Equity Premium Income ETF",
                aum=aum,
                as_of_date=as_of_date,
                strategy_note=note,
                visibility="Largest opaque flow",
                source_url=JEPQ_DAILY_HOLDINGS_URL,
            )
            _write_json_cache(cache_name, _serialize_flow_reference(reference))
            return reference
        except Exception:
            cached = _read_json_cache(cache_name)
            if not cached:
                return None
            return _deserialize_flow_reference(cached)

else:

    def load_jheqx_quarterly_snapshots(limit: int = 12) -> list[QuarterlyCollarSnapshot]:
        raise RuntimeError("streamlit is not installed")


    def load_latest_monthly_snapshot() -> QuarterlyCollarSnapshot | None:
        raise RuntimeError("streamlit is not installed")


    def load_market_prices(symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        raise RuntimeError("streamlit is not installed")


    def load_qqqi_public_snapshot() -> OverlaySnapshot | None:
        raise RuntimeError("streamlit is not installed")


    def load_jepq_flow_reference() -> FlowReference | None:
        raise RuntimeError("streamlit is not installed")


def _merge_public_snapshots(
    quarterly_snapshots: list[QuarterlyCollarSnapshot],
    monthly_snapshot: QuarterlyCollarSnapshot | None,
) -> list[QuarterlyCollarSnapshot]:
    merged: dict[str, QuarterlyCollarSnapshot] = {
        snapshot.quarter_label: snapshot for snapshot in quarterly_snapshots
    }
    if monthly_snapshot is not None:
        existing = merged.get(monthly_snapshot.quarter_label)
        if existing is None or monthly_snapshot.as_of_date >= existing.as_of_date:
            merged[monthly_snapshot.quarter_label] = monthly_snapshot
    return sorted(merged.values(), key=lambda item: item.as_of_date, reverse=True)


def _extend_snapshots_with_estimate(
    snapshots: list[QuarterlyCollarSnapshot],
    spx_prices: list[dict[str, Any]],
    latest_market_date: date,
) -> list[QuarterlyCollarSnapshot]:
    if not snapshots:
        return []

    enriched: list[QuarterlyCollarSnapshot] = []
    for snapshot in snapshots:
        quarter_start = _quarter_start(snapshot.as_of_date)
        quarter_end = _quarter_end(snapshot.as_of_date)
        hedge_start = _first_market_day(spx_prices, quarter_start, quarter_end)
        spx_level = _find_close_on_or_before(spx_prices, snapshot.as_of_date)
        enriched.append(
            QuarterlyCollarSnapshot(
                quarter_label=snapshot.quarter_label,
                hedge_start=hedge_start,
                hedge_end=quarter_end,
                as_of_date=snapshot.as_of_date,
                spx_level=spx_level,
                long_put=snapshot.long_put,
                short_put=snapshot.short_put,
                short_call=snapshot.short_call,
                source_url=snapshot.source_url,
                source_kind=snapshot.source_kind,
                is_disclosed=True,
                is_estimated=False,
            )
        )

    latest_disclosed = enriched[0]
    next_start = _next_quarter_start(latest_disclosed.as_of_date)
    if latest_market_date >= next_start:
        estimated_end = latest_market_date
        estimated_start = _first_market_day(spx_prices, next_start, _quarter_end(latest_market_date))
        enriched.insert(
            0,
            QuarterlyCollarSnapshot(
                quarter_label=_quarter_label(latest_market_date),
                hedge_start=estimated_start,
                hedge_end=estimated_end,
                as_of_date=latest_disclosed.as_of_date,
                spx_level=latest_disclosed.spx_level,
                long_put=latest_disclosed.long_put,
                short_put=latest_disclosed.short_put,
                short_call=latest_disclosed.short_call,
                source_url=latest_disclosed.source_url,
                source_kind="Carry-forward estimate",
                is_disclosed=False,
                is_estimated=True,
            )
        )
    return enriched


def build_overlay_series(
    snapshots: list[Any],
    market_prices: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if not snapshots:
        return {"price_rows": [], "option_rows": [], "combined_rows": []}

    underlying_symbol = getattr(snapshots[0], "underlying_symbol", "^GSPC")
    market_series_name = "NDX Index" if underlying_symbol == "^NDX" else "SPX Index"
    price_rows = [
        {
            "date": row["date"].isoformat(),
            "value": float(row["close"]),
            "series": market_series_name,
            "status": "Market",
        }
        for row in market_prices
    ]

    ordered_snapshots = sorted(snapshots, key=lambda item: (item.hedge_start, item.hedge_end))
    option_rows: list[dict[str, Any]] = []
    series_names: list[str] = []
    snapshot_entries: list[dict[str, float]] = []
    for snapshot in ordered_snapshots:
        entries: dict[str, float] = {}
        legs = getattr(snapshot, "legs", ())
        if legs:
            for leg in legs:
                entries[leg.series] = leg.strike
                if leg.series not in series_names:
                    series_names.append(leg.series)
        else:
            for series_name, attr_name in (
                ("Long Put", "long_put"),
                ("Short Put", "short_put"),
                ("Short Call", "short_call"),
            ):
                strike_value = getattr(snapshot, attr_name, None)
                if strike_value is None:
                    continue
                entries[series_name] = float(strike_value)
                if series_name not in series_names:
                    series_names.append(series_name)
        snapshot_entries.append(entries)

    for series_name in series_names:
        last_snapshot: Any | None = None
        last_value: float | None = None
        for snapshot, entries in zip(ordered_snapshots, snapshot_entries):
            strike_value = entries.get(series_name)
            if strike_value is None:
                continue
            last_snapshot = snapshot
            last_value = strike_value
            option_rows.append(
                {
                    "date": snapshot.hedge_start.isoformat(),
                    "value": strike_value,
                    "series": series_name,
                    "status": "Estimated" if snapshot.is_estimated else "Disclosed",
                }
            )
        if last_snapshot is not None and last_value is not None:
            option_rows.append(
                {
                    "date": last_snapshot.hedge_end.isoformat(),
                    "value": last_value,
                    "series": series_name,
                    "status": "Estimated" if last_snapshot.is_estimated else "Disclosed",
                }
            )

    option_rows.sort(key=lambda row: (row["series"], row["date"]))
    combined_rows = sorted(price_rows + option_rows, key=lambda row: (row["date"], row["series"]))
    return {
        "price_rows": price_rows,
        "option_rows": option_rows,
        "combined_rows": combined_rows,
    }


def _build_overlay_chart_spec(
    y_title: str,
    height: int = 420,
    series_colors: dict[str, str] | None = None,
) -> dict[str, Any]:
    tooltip = [
        {"field": "date", "type": "temporal", "title": "Date"},
        {"field": "series", "type": "nominal", "title": "Series"},
        {"field": "value", "type": "quantitative", "title": "Value", "format": ",.2f"},
        {"field": "status", "type": "nominal", "title": "Status"},
    ]
    if series_colors is None:
        series_colors = {
            "SPX Index": "#1f4e79",
            "Short Put": "#d62728",
            "Long Put": "#7f3fbf",
            "Short Call": "#2ca02c",
        }
    color_scale = {
        "domain": list(series_colors.keys()),
        "range": list(series_colors.values()),
    }
    layers = [
        {
            "transform": [{"filter": "datum.status === 'Market'"}],
            "mark": {"type": "line", "point": False, "strokeWidth": 2.8},
            "encoding": {
                "x": {
                    "field": "date",
                    "type": "temporal",
                    "title": "",
                    "axis": {"labelAngle": -45},
                },
                "y": {"field": "value", "type": "quantitative", "title": y_title},
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "scale": color_scale,
                    "legend": {"orient": "bottom"},
                },
                "tooltip": tooltip,
            },
        },
        {
            "transform": [{"filter": "datum.status === 'Current'"}],
            "mark": {"type": "line", "point": False, "strokeWidth": 2.2, "strokeDash": [6, 4]},
            "encoding": {
                "x": {"field": "date", "type": "temporal", "title": ""},
                "y": {"field": "value", "type": "quantitative", "title": y_title},
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "scale": color_scale,
                    "legend": {"orient": "bottom"},
                },
                "tooltip": tooltip,
            },
        },
        {
            "transform": [{"filter": "datum.status !== 'Market' && datum.status !== 'Current'"}],
            "mark": {"type": "line", "point": False, "interpolate": "step-after", "strokeWidth": 2.5},
            "encoding": {
                "x": {"field": "date", "type": "temporal", "title": ""},
                "y": {"field": "value", "type": "quantitative", "title": y_title},
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "scale": color_scale,
                    "legend": {"orient": "bottom"},
                },
                "strokeDash": {
                    "field": "status",
                    "type": "nominal",
                    "scale": {"domain": ["Disclosed", "Estimated"], "range": [[1, 0], [6, 4]]},
                    "legend": {"orient": "bottom"},
                },
                "tooltip": tooltip,
            },
        },
    ]

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": height,
        "layer": layers,
    }


def _snapshot_table_rows(snapshots: list[QuarterlyCollarSnapshot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        rows.append(
            {
                "Quarter": snapshot.quarter_label,
                "Hedge Start": snapshot.hedge_start.isoformat(),
                "Hedge End": snapshot.hedge_end.isoformat(),
                "As Of": snapshot.as_of_date.isoformat(),
                "SPX Index": round(snapshot.spx_level, 2) if snapshot.spx_level is not None else None,
                "Long Put": snapshot.long_put,
                "Short Put": snapshot.short_put,
                "Short Call": snapshot.short_call,
                "State": "Estimated" if snapshot.is_estimated else "Disclosed",
                "Source": snapshot.source_kind,
                "URL": snapshot.source_url,
            }
        )
    return rows


def _format_aum(aum: float | None) -> str:
    if aum is None:
        return "-"
    if aum >= 1_000_000_000:
        return f"${aum / 1_000_000_000:.2f}B"
    return f"${aum / 1_000_000:.1f}M"


def _overlay_series_colors(market_series_name: str, option_series: list[str]) -> dict[str, str]:
    colors = {
        market_series_name: "#1f4e79",
        "Current NDX": "#1f4e79",
        "Long Put": "#7f3fbf",
        "Short Put": "#d97706",
        "Short Call": "#d62728",
        "Long Call": "#2ca02c",
    }
    resolved: dict[str, str] = {market_series_name: colors[market_series_name]}
    short_call_palette = ["#d62728", "#ff7f0e", "#b22222"]
    long_call_palette = ["#2ca02c", "#17becf", "#3cb371"]
    short_put_palette = ["#d97706", "#bc6c25", "#8c564b"]
    long_put_palette = ["#7f3fbf", "#9467bd", "#6a3d9a"]
    short_call_index = long_call_index = short_put_index = long_put_index = 0

    for series_name in option_series:
        if series_name in resolved:
            continue
        if series_name.startswith("Short Call"):
            resolved[series_name] = short_call_palette[min(short_call_index, len(short_call_palette) - 1)]
            short_call_index += 1
        elif series_name.startswith("Long Call"):
            resolved[series_name] = long_call_palette[min(long_call_index, len(long_call_palette) - 1)]
            long_call_index += 1
        elif series_name.startswith("Short Put"):
            resolved[series_name] = short_put_palette[min(short_put_index, len(short_put_palette) - 1)]
            short_put_index += 1
        elif series_name.startswith("Long Put"):
            resolved[series_name] = long_put_palette[min(long_put_index, len(long_put_palette) - 1)]
            long_put_index += 1
        else:
            resolved[series_name] = "#666666"
    return resolved


def _qqqi_leg_table_rows(snapshot: OverlaySnapshot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for leg in snapshot.legs:
        rows.append(
            {
                "Series": leg.series,
                "Strike": leg.strike,
                "Expiry": leg.expiry_date.isoformat() if leg.expiry_date else None,
                "Contracts": leg.contracts,
                "Source": snapshot.source_kind,
                "URL": snapshot.source_url,
            }
        )
    return rows


def _extend_overlay_with_current_level(
    overlay_rows: list[dict[str, Any]],
    snapshot: OverlaySnapshot,
    series_name: str,
) -> list[dict[str, Any]]:
    if snapshot.underlying_level is None or snapshot.hedge_end <= snapshot.as_of_date:
        return overlay_rows
    rows = list(overlay_rows)
    rows.append(
        {
            "date": snapshot.as_of_date.isoformat(),
            "value": snapshot.underlying_level,
            "series": series_name,
            "status": "Current",
        }
    )
    rows.append(
        {
            "date": snapshot.hedge_end.isoformat(),
            "value": snapshot.underlying_level,
            "series": series_name,
            "status": "Current",
        }
    )
    return sorted(rows, key=lambda row: (row["date"], row["series"]))


def render_nasdaq_anchor_section() -> None:
    st.divider()
    st.header("Nasdaq Anchor")
    st.caption(
        "Track the largest public Nasdaq overlay capital alongside JHEQX. QQQI is charted from current issuer holdings, "
        "while JEPQ is shown as a large opaque ELN-based flow reference."
    )

    qqqi_snapshot = load_qqqi_public_snapshot()
    jepq_reference = load_jepq_flow_reference()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("QQQI")
        if qqqi_snapshot is None:
            st.warning("QQQI public holdings metadata could not be loaded.")
        else:
            qqqi_aum = None
            try:
                qqqi_csv = _request_text(NEOS_QQQI_HOLDINGS_CSV_URL, NEOS_HEADERS)
                qqqi_rows = list(csv.DictReader(StringIO(qqqi_csv)))
                if qqqi_rows:
                    qqqi_aum = _parse_optional_float(qqqi_rows[0].get("NetAssets"))
            except Exception:
                qqqi_aum = None
            st.metric("Public AUM", _format_aum(qqqi_aum))
            st.caption(
                f"Chartable spread/overlay | Latest public holdings: {qqqi_snapshot.as_of_date.isoformat()} | "
                f"{len(qqqi_snapshot.legs)} disclosed NDX option leg(s)"
            )
    with c2:
        st.subheader("JEPQ")
        if jepq_reference is None:
            st.warning("JEPQ metadata unavailable from the official holdings source.")
        else:
            st.metric("Estimated AUM", _format_aum(jepq_reference.aum))
            if jepq_reference.as_of_date is not None:
                st.caption(
                    f"{jepq_reference.visibility} | As of {jepq_reference.as_of_date.isoformat()}"
                )
            else:
                st.caption(jepq_reference.visibility)

    st.subheader("QQQI NDX Overlay")
    if qqqi_snapshot is None:
        st.info("QQQI could not be loaded from the issuer site, so no Nasdaq anchor chart is shown.")
    elif not qqqi_snapshot.legs:
        st.warning(
            "QQQI metadata loaded, but no public NDX option legs were parsed. The app does not invent synthetic levels."
        )
    else:
        ndx_start = max(qqqi_snapshot.as_of_date - timedelta(days=45), date(2024, 1, 1))
        ndx_end = min(date.today(), qqqi_snapshot.as_of_date)
        try:
            ndx_prices = load_market_prices("^NDX", ndx_start, ndx_end)
        except Exception as exc:
            st.error(f"Failed to load NDX price data: {exc}")
            ndx_prices = []

        if ndx_prices:
            enriched_snapshot = replace(
                qqqi_snapshot,
                underlying_level=_find_close_on_or_before(ndx_prices, qqqi_snapshot.as_of_date),
            )
            overlay = build_overlay_series([enriched_snapshot], ndx_prices)
            combined_rows = _extend_overlay_with_current_level(
                overlay["combined_rows"],
                enriched_snapshot,
                "Current NDX",
            )
            option_series = [leg.series for leg in enriched_snapshot.legs] + ["Current NDX"]
            series_colors = _overlay_series_colors("NDX Index", option_series)
            st.vega_lite_chart(
                combined_rows,
                _build_overlay_chart_spec(
                    y_title="NDX Level / Option Strike",
                    height=360,
                    series_colors=series_colors,
                ),
                use_container_width=True,
            )
            st.caption(
                "QQQI is shown as a current-cycle NDX call spread anchor. The solid blue line is recent NDX history, the "
                "dashed blue line extends the current NDX level through option expiry, and the red/orange lines are the "
                "currently disclosed call strikes."
            )
            st.dataframe(_qqqi_leg_table_rows(enriched_snapshot), use_container_width=True, hide_index=True)

    st.subheader("JEPQ Reference Card")
    if jepq_reference is None:
        st.warning("JEPQ metadata unavailable.")
    else:
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Ticker", jepq_reference.ticker)
        with r2:
            st.metric("Estimated AUM", _format_aum(jepq_reference.aum))
        with r3:
            st.metric("Visibility", jepq_reference.visibility)
        st.caption(jepq_reference.strategy_note)
        st.caption(f"Official source: {jepq_reference.source_url}")


def render_jheqx_collar_page() -> None:
    if st is None:
        raise RuntimeError("streamlit is not installed")

    st.set_page_config(page_title="JHEQX Collar Hedge", layout="wide")
    st.title("JHEQX Collar Hedge Quarterly Overlay")
    st.caption(
        "Reconstruct the JHEQX SPX collar from public SEC N-PORT filings, and refresh the latest quarter "
        "from J.P. Morgan's public monthly holdings PDF when it is available."
    )
    st.caption(f"History window is fixed to the latest {DEFAULT_VISIBLE_QUARTERS} quarters.")
    limit = DEFAULT_VISIBLE_QUARTERS
    try:
        quarterly_snapshots = load_jheqx_quarterly_snapshots(limit=limit + 2)
    except Exception as exc:
        st.error(f"Failed to load JHEQX filing data: {exc}")
        render_nasdaq_anchor_section()
        return

    monthly_snapshot = load_latest_monthly_snapshot()
    snapshots = _merge_public_snapshots(quarterly_snapshots, monthly_snapshot)

    market_start = _quarter_start(snapshots[-1].as_of_date) - timedelta(days=10)
    market_end = date.today()

    try:
        spx_prices = load_market_prices("^GSPC", market_start, market_end)
    except Exception as exc:
        st.error(f"Failed to load SPX price data: {exc}")
        render_nasdaq_anchor_section()
        return

    latest_market_date = spx_prices[-1]["date"]
    visible_snapshots = _extend_snapshots_with_estimate(snapshots, spx_prices, latest_market_date)[: limit + 1]
    overlay = build_overlay_series(visible_snapshots, spx_prices)

    latest_snapshot = next((item for item in visible_snapshots if not item.is_estimated), None)
    estimated_snapshot = next((item for item in visible_snapshots if item.is_estimated), None)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Latest public quarter", latest_snapshot.quarter_label if latest_snapshot else "-")
    with m2:
        st.metric("Long Put", f"{latest_snapshot.long_put:,.0f}" if latest_snapshot and latest_snapshot.long_put else "-")
    with m3:
        st.metric("Short Put", f"{latest_snapshot.short_put:,.0f}" if latest_snapshot and latest_snapshot.short_put else "-")
    with m4:
        st.metric("Short Call", f"{latest_snapshot.short_call:,.0f}" if latest_snapshot and latest_snapshot.short_call else "-")

    if latest_snapshot is not None:
        st.caption(
            f"Latest public source: {latest_snapshot.source_kind} as of {latest_snapshot.as_of_date.isoformat()}."
        )

    if estimated_snapshot is not None:
        st.warning(
            f"{estimated_snapshot.quarter_label} has no public filing yet. "
            f"It is displayed as a carry-forward estimate from the last public snapshot dated "
            f"{estimated_snapshot.as_of_date.isoformat()}."
        )
    elif monthly_snapshot is None and PdfReader is None:
        st.info("Install `pypdf` to refresh the latest quarter from J.P. Morgan's public monthly holdings PDF.")

    st.subheader("Price And Collar Overlay")
    st.vega_lite_chart(
        overlay["combined_rows"],
        _build_overlay_chart_spec(
            y_title="SPX Level / Option Strike",
            height=420,
        ),
        use_container_width=True,
    )

    st.subheader("Quarterly Source Table")
    st.dataframe(_snapshot_table_rows(visible_snapshots), use_container_width=True, hide_index=True)
    st.caption(
        "All lines are shown on the native SPX scale. SPX Index is the market path, and the collar levels are drawn as "
        "step lines so each quarter stays flat until the next hedge reset. When the most recent quarter is available only "
        "from the monthly holdings PDF, that public monthly source overrides the SEC carry-forward estimate."
    )
    render_nasdaq_anchor_section()


def main() -> None:
    render_jheqx_collar_page()


if __name__ == "__main__":
    main()

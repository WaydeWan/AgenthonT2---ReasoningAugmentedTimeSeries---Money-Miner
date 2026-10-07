"""Dated source ingestion and literal SEP tables, without model calls.

SEP participant projections are policy assessments, not Treasury yield forecasts.
Their cross-release changes are features; no change is called market surprise.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from statistics import median

MAX_DOCUMENTS = 80
MAX_DOCUMENT_BYTES = 2_000_000
MAX_CORPUS_BYTES = 12_000_000
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
MONTHS = r"January|February|March|April|May|June|July|August|September|October|November|December"
VARIABLES = {"Change in real GDP": "gdp_growth", "Unemployment rate": "unemployment",
             "PCE inflation": "pce_inflation", "Core PCE inflation": "core_pce_inflation",
             "Federal funds rate": "policy_rate"}
RATE_LITERAL = r"(?:\d+-\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"


class EvidenceError(ValueError):
    pass


def date_value(value):
    if not isinstance(value, str):
        raise EvidenceError("invalid_date")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.date()
    except ValueError:
        raise EvidenceError("invalid_date") from None


def load_corpus(text_dir, asof):
    """Read only index fields needed for sources; no card title/note/answer hints."""
    root = Path(text_dir).resolve()
    cutoff = date_value(asof)
    index_file = root / "corpus_index.json"
    if index_file.stat().st_size > 1_000_000:
        raise EvidenceError("index_budget")
    index = json.loads(index_file.read_text(encoding="utf-8-sig"))
    rows = index.get("documents")
    if not isinstance(rows, list) or len(rows) > MAX_DOCUMENTS:
        raise EvidenceError("document_count_budget")
    documents, seen, total, skipped = [], set(), 0, []
    for row in rows:
        if not isinstance(row, dict):
            raise EvidenceError("invalid_document_record")
        stamp = row.get("timestamp", row.get("date"))
        if date_value(stamp) > cutoff:
            skipped.append({"reason": "future_document"})
            continue
        name, identity = row.get("file"), row.get("doc_id")
        if not isinstance(name, str) or not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise EvidenceError("invalid_document_identity")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".txt":
            raise EvidenceError("unsafe_document_path")
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise EvidenceError("document_budget")
        raw = path.read_bytes()
        total += len(raw)
        if total > MAX_CORPUS_BYTES:
            raise EvidenceError("corpus_budget")
        text = raw.decode("utf-8-sig")
        seen.add(identity)
        documents.append({"doc_id": identity, "published_at": stamp,
                          "source": str(row.get("source", "")), "doc_type": str(row.get("doc_type", "")),
                          "text": text, "sha256": hashlib.sha256(raw).hexdigest()})
    return sorted(documents, key=lambda d: (date_value(d["published_at"]), d["doc_id"])), skipped


def _cell(value):
    value = value.strip()
    if not value or value in ("-", "—", "…", "NA", "N/A"):
        return None
    match = re.fullmatch(rf"({NUMBER})\s*(?:to|[–—])\s*({NUMBER})", value)
    if match:
        low, high = map(float, match.groups())
    elif re.fullmatch(NUMBER, value):
        low = high = float(value)
    else:
        return None
    if low > high or max(abs(low), abs(high)) > 100:
        return None
    return low, high


def _header(value):
    match = re.fullmatch(r"(Median|Central\s+[Tt]endency|Range)\s*(?:\[\d+\])?\s*(\d{4}|Longer run)", value.strip())
    if not match:
        return None
    statistic = {"Median": "median", "Range": "range"}.get(match[1], "central_tendency")
    return statistic, "longer_run" if match[2] == "Longer run" else match[2]


def parse_sep_tables(document):
    """Literal table parsing; retain row/header citations and missing cells."""
    lines = document["text"].splitlines()
    facts, dots = [], []
    for line_index, line in enumerate(lines):
        columns = [cell.strip() for cell in line.split("|")]
        if columns[0] == "Variable" and len(columns) >= 3:
            headings = [_header(cell) for cell in columns[1:]]
            if any(value is None for value in headings):
                continue
            variable = None
            for row_index in range(line_index + 1, len(lines)):
                raw = lines[row_index]
                if not raw.strip():
                    break
                cells = [cell.strip() for cell in raw.split("|")]
                if len(cells) == 1 and cells[0].startswith("Memo:"):
                    continue
                if len(cells) != len(columns):
                    break
                label = re.sub(r"\s*\[\d+\]", "", cells[0]).strip()
                prior = bool(re.fullmatch(rf"(?:{MONTHS}) projection", label))
                if not prior:
                    variable = VARIABLES.get(label)
                if variable is None:
                    continue
                for column, (value, header) in enumerate(zip(cells[1:], headings), start=1):
                    parsed = _cell(value)
                    if parsed is None:
                        continue
                    low, high = parsed
                    facts.append({"doc_id": document["doc_id"], "published_at": document["published_at"],
                                  "variable": variable, "statistic": header[0], "period": header[1],
                                  "vintage": "prior_reported" if prior else "current", "vintage_label": label,
                                  "lower": low, "upper": high, "unit": "percent",
                                  "source_line": row_index + 1, "source_column": column,
                                  "header_quote": line, "row_quote": raw, "cell_quote": value})
        if columns[0] == "Midpoint of target range or target level (Percent)" and len(columns) >= 3:
            periods = ["longer_run" if col == "Longer run" else col for col in columns[1:]]
            if any(not re.fullmatch(r"\d{4}|longer_run", period) for period in periods):
                continue
            values, sources, valid = [[] for _ in periods], [], True
            for row_index in range(line_index + 1, len(lines)):
                raw = lines[row_index]
                if not raw.strip():
                    break
                cells = [cell.strip() for cell in raw.split("|")]
                if len(cells) != len(columns) or not re.fullmatch(NUMBER, cells[0]):
                    valid = False
                    break
                rate = float(cells[0])
                if not -20 <= rate <= 100:
                    valid = False
                    break
                for index, cell in enumerate(cells[1:]):
                    if cell and (not cell.isdigit() or int(cell) > 100):
                        valid = False
                        break
                    values[index].extend([rate] * int(cell or 0))
                if not valid:
                    break
                sources.append({"source_line": row_index + 1, "row_quote": raw})
            if not valid:
                continue
            for period, numbers in zip(periods, values):
                if not numbers or len(numbers) > 100:
                    continue
                dots.append({"doc_id": document["doc_id"], "published_at": document["published_at"],
                             "variable": "policy_rate", "period": period,
                             "statistic": "participant_dot_distribution", "unit": "percent",
                             "participant_count": len(numbers), "median": median(numbers),
                             "minimum": min(numbers), "maximum": max(numbers),
                             "values": sorted(numbers), "header_quote": line, "rows": sources,
                             "boundary": "Participant policy assessments; not market pricing or a yield distribution."})
    return facts, dots


def sep_features(documents, asof):
    """Latest SEP features by calendar offset; missing values stay explicit."""
    year = date_value(asof).year
    parsed = []
    for document in documents:
        if date_value(document["published_at"]) > date_value(asof):
            raise EvidenceError("future_document")
        facts, dots = parse_sep_tables(document)
        if facts or dots:
            parsed.append((document, facts, dots))
    if not parsed:
        return {"features": {}, "facts": [], "dots": [], "latest_document": None}
    document, facts, dots = max(parsed, key=lambda row: date_value(row[0]["published_at"]))
    features = {}
    lookup = {(f["variable"], f["statistic"], f["period"], f["vintage"]): f for f in facts}
    for variable in VARIABLES.values():
        for offset in range(3):
            period = str(year + offset)
            for statistic in ("median", "central_tendency"):
                current = lookup.get((variable, statistic, period, "current"))
                if current is None:
                    continue
                center = (current["lower"] + current["upper"]) / 2
                name = f"sep_{variable}_{statistic}_y{offset}"
                features[name + "_level"] = center
                features[name + "_width"] = current["upper"] - current["lower"]
                previous = lookup.get((variable, statistic, period, "prior_reported"))
                if previous is not None:
                    features[name + "_revision"] = center - (previous["lower"] + previous["upper"]) / 2
    for dot in dots:
        if dot["period"].isdigit() and 0 <= int(dot["period"]) - year <= 2:
            offset = int(dot["period"]) - year
            features[f"sep_policy_dot_y{offset}_median"] = dot["median"]
            features[f"sep_policy_dot_y{offset}_range"] = dot["maximum"] - dot["minimum"]
    features["sep_age_calendar_days"] = (date_value(asof) - date_value(document["published_at"])).days
    return {"features": features, "facts": facts, "dots": dots, "latest_document": document["doc_id"]}


def policy_target_features(documents, asof):
    """Narrow literal current Fed target, not every percentage in a statement."""
    records = []
    for doc in documents:
        if date_value(doc["published_at"]) > date_value(asof):
            raise EvidenceError("future_document")
        if doc.get("doc_type") != "fomc_statement":
            continue
        for paragraph in re.split(r"\n\s*\n", doc["text"]):
            # Only an explicitly completed Committee decision/current reaffirmation.
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph):
                normalized = sentence.replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4")
                action = re.search(r"\bCommittee (?:decided to (?:maintain|raise|lower)|today reaffirmed its view that the current)\b", normalized)
                if not action or re.search(r"\b(if|would|may|might|unless|until|when|not)\b", normalized, re.I):
                    continue
                match = re.search(rf"(?:target range for the federal funds rate (?:at|to)|the current)\s+({RATE_LITERAL})\s+to\s+({RATE_LITERAL})\s+percent", normalized)
                if not match:
                    continue
                def value(literal):
                    if "-" in literal:
                        integer, fraction = literal.split("-", 1)
                        return float(int(integer) + Fraction(fraction))
                    return float(Fraction(literal))
                low, high = value(match[1]), value(match[2])
                if not 0 <= low <= high <= 30:
                    continue
                records.append({"doc_id": doc["doc_id"], "published_at": doc["published_at"],
                                "lower": low, "upper": high, "unit": "percent", "quote": sentence,
                                "parent_quote": paragraph})
                break
    by_date = {}
    for record in records:
        day = str(date_value(record["published_at"]))
        if day in by_date and (by_date[day]["lower"], by_date[day]["upper"]) != (record["lower"], record["upper"]):
            return {"features": {}, "facts": records, "reason": "conflicting_policy_targets"}
        by_date[day] = record
    ordered = [by_date[day] for day in sorted(by_date)]
    if not ordered:
        return {"features": {}, "facts": [], "reason": "no_explicit_current_target"}
    latest = ordered[-1]
    center = (latest["lower"] + latest["upper"]) / 2
    features = {"policy_current_target_midpoint": center,
                "policy_current_target_bandwidth": latest["upper"] - latest["lower"],
                "policy_statement_age_days": (date_value(asof) - date_value(latest["published_at"])).days}
    if len(ordered) >= 2:
        previous = ordered[-2]
        features["policy_target_change_from_previous_statement"] = center - (previous["lower"] + previous["upper"]) / 2
    return {"features": features, "facts": ordered, "reason": None}

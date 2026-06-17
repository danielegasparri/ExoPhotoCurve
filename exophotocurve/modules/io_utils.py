"""Input/output helpers for ASCII photometric tables."""

from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd

from .constants import NONE_COL


def clean_column_names(columns: List[object]) -> List[str]:
    """Return clean, unique column names."""
    cleaned: List[str] = []
    seen: Dict[str, int] = {}

    for i, col in enumerate(columns):
        name = str(col).strip()

        if name == "" or name.lower().startswith("unnamed"):
            name = f"col_{i}"

        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0

        cleaned.append(name)

    return cleaned


def read_ascii_table(path: str, delimiter_choice: str, header: bool) -> pd.DataFrame:
    """Read an ASCII table with flexible delimiter options.

    AstroImageJ tables often use '#Label' as the first header field. In that
    case the leading '#' is part of the header line, not a normal comment.
    This reader handles both the original AIJ format and the manually edited
    version where '#Label' has already been changed to 'Label'.
    """
    delimiter_map = {
        "Auto": None,
        "Whitespace": r"\s+",
        "Tab": "\t",
        "Comma": ",",
        "Semicolon": ";",
    }

    sep = delimiter_map.get(delimiter_choice, None)
    header_row = 0 if header else None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.read().splitlines()

    non_empty_lines = [line for line in raw_lines if line.strip()]
    if not non_empty_lines:
        raise ValueError("The selected file is empty.")

    first_line = non_empty_lines[0]
    is_aij_table = first_line.startswith("#Label") or first_line.startswith("Label")

    if is_aij_table and header:
        cleaned_lines: List[str] = []
        header_done = False

        for line in raw_lines:
            if not line.strip():
                continue

            if not header_done:
                if line.startswith("#Label"):
                    line = line.replace("#Label", "Label", 1)
                cleaned_lines.append(line)
                header_done = True
                continue

            # After the header, ignore possible real comment lines.
            if line.lstrip().startswith("#"):
                continue

            cleaned_lines.append(line)

        df = pd.read_csv(
            io.StringIO("\n".join(cleaned_lines)),
            sep="\t" if delimiter_choice == "Auto" else sep,
            engine="python",
            header=0,
            skip_blank_lines=True,
            na_values=["NaN", "nan", "--", "INDEF", "Infinity", "-Infinity"],
        )

    else:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                comment="#",
                header=header_row,
                skip_blank_lines=True,
                na_values=["NaN", "nan", "--", "INDEF", "Infinity", "-Infinity"],
            )
        except Exception:
            # Fallback for many whitespace-separated ASCII tables.
            df = pd.read_csv(
                path,
                sep=r"\s+",
                engine="python",
                comment="#",
                header=header_row,
                skip_blank_lines=True,
                na_values=["NaN", "nan", "--", "INDEF", "Infinity", "-Infinity"],
            )

        if df.shape[1] <= 1 and delimiter_choice == "Auto":
            df = pd.read_csv(
                path,
                sep=r"\s+",
                engine="python",
                comment="#",
                header=header_row,
                skip_blank_lines=True,
                na_values=["NaN", "nan", "--", "INDEF", "Infinity", "-Infinity"],
            )

    df.columns = clean_column_names(list(df.columns))

    # Convert numeric columns, keeping string columns such as Label untouched.
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted

    return df


def numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that contain numeric data."""
    cols: List[str] = []

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            cols.append(col)

    return cols


def guess_column(columns: List[str], keywords: List[str]) -> str:
    """Guess a column using a list of case-insensitive keywords."""
    lower_map = {col: col.lower() for col in columns}

    # First try exact matches, then substring matches. This avoids selecting
    # rel_flux_T1 before rel_flux_T1_dfn when both are present.
    for key in keywords:
        key_l = key.lower()
        for col, col_l in lower_map.items():
            if col_l == key_l:
                return col

    for key in keywords:
        key_l = key.lower()
        for col, col_l in lower_map.items():
            if key_l in col_l:
                return col

    return NONE_COL

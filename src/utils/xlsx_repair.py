"""Repair malformed .xlsx document-properties timestamps.

Several official government/Fed data workbooks (observed in Philadelphia
Fed SPF/Livingston/ATSIX exports, which are generated via SAS) write
``docProps/core.xml`` timestamps with a space instead of a zero-pad digit,
e.g. ``2026-05-13T 2:35:20-04:00`` instead of ``2026-05-13T02:35:20-04:00``.
This is invalid W3CDTF and makes openpyxl's document-properties parser
raise ``TypeError: expected <class 'datetime.datetime'>`` before it ever
gets to reading worksheet data we actually want.

This module patches only the malformed timestamp text inside the zip
in-place; it never touches worksheet content, so it cannot silently alter
data.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

_BAD_TIME_RE = re.compile(r"T\s(\d:\d{2}:\d{2})")
_CORE_XML_NAME = "docProps/core.xml"


def repair_xlsx_in_place(path: Path) -> bool:
    """Fix malformed W3CDTF timestamps in ``docProps/core.xml`` if present.

    Returns True if a repair was applied, False if the file needed no fix
    (including if it has no core.xml, e.g. non-xlsx files).
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if _CORE_XML_NAME not in zf.namelist():
                return False
            core_xml = zf.read(_CORE_XML_NAME).decode("utf-8")
    except zipfile.BadZipFile:
        return False

    if not _BAD_TIME_RE.search(core_xml):
        return False

    fixed_xml = _BAD_TIME_RE.sub(lambda m: f"T0{m.group(1)}", core_xml)
    logger.info("repairing malformed core.xml timestamp in %s", path)

    tmp_path = path.with_suffix(path.suffix + ".repair.tmp")
    with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == _CORE_XML_NAME:
                data = fixed_xml.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(str(tmp_path), str(path))
    return True

from __future__ import annotations

import copy
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Iterable
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, File, Query, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from urllib.parse import quote

import db as guide_db


ROOT = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / "path"
IMAGES_DIR = ROOT / "图片"
ASSETS_DIR = ROOT / "static"
TEMPLATES_DIR = ROOT / "templates"
DB_PATH = ROOT / "data.db"
DOCS_DIR = ROOT / "library_files"
# 站点基础地址（用于生成邀请/复制链接）
BASE_URL ="http://127.0.0.1:8000"

# 审计日志（按 IP 记录关键操作）
LOG_DIR = ROOT / "logs"
AUDIT_LOG_PATH = LOG_DIR / "audit.log"


SETTING_SHOES_TIPS = "guide_shoes_tips"
SETTING_TICKET_TIPS = "guide_ticket_tips"
DEFAULT_SHOES_TIPS = [
    "新手优先考虑舒适：包紧但不痛",
    "脚后跟不要空 🙅‍♀️",
    "量好脚长再选鞋：不同品牌尺码差异大",
    "建议线下试鞋：香蕉、香港沙木尼等",
    "香港沙木尼满 ¥1500 可享 75 折",
]
DEFAULT_TICKET_TIPS = [
    "美团：香蕉工作日闲时 ¥69",
    "cika exchange 小程序：与岩友购买实惠次卡",
    "圳惠生活小程序：香蕉攀岩（仅限工作日）",
    "南山文体通：抢 100-40 消费券",
    "福田文体通",
]


app = FastAPI(title="深圳爬墙区新手村指南", version="0.2.0")
ENABLE_RELOAD = os.getenv("ENABLE_GUIDE_RELOAD", "0") == "1"
ENABLE_EDIT = True

# 哪些工作表的图片列用作“预览图”
_PREVIEW_IMAGE_SHEETS = {"其他设备", "攀岩鞋"}

# Static assets (css/js)
if ASSETS_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(ASSETS_DIR)), name="static")

# User-provided images
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _setup_audit_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("audit")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    # 按天切割日志，保留最近 90 天
    # 生成文件：audit.log（当天）、audit.log.2026-02-10 等
    handler = TimedRotatingFileHandler(
        str(AUDIT_LOG_PATH),
        when="midnight",
        interval=1,
        backupCount=90,
        encoding="utf-8",
        utc=False,
    )
    handler.suffix = "%Y-%m-%d"
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


AUDIT_LOGGER = _setup_audit_logger()

# ============ IP 封禁 & IP-昵称关联 ============
BANNED_IPS_PATH = LOG_DIR / "banned_ips.json"
IP_NICKNAMES_PATH = LOG_DIR / "ip_nicknames.json"

_ban_lock = threading.Lock()
_nick_lock = threading.Lock()

# 内存中的 404 计数器：{ ip: [timestamp, timestamp, ...] }
_404_counter: dict[str, list[float]] = {}
_404_counter_lock = threading.Lock()

# 封禁窗口与阈值
_BAN_WINDOW = 10       # 秒
_BAN_THRESHOLD = 10    # 次数


def _load_json_file(path: Path) -> dict:
    """安全读取 JSON 文件，不存在/损坏则返回空 dict。"""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_json_file(path: Path, data: dict) -> None:
    """原子写入 JSON 文件（先写临时文件再 rename）。"""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ---- 封禁相关 ----

def _load_banned_ips() -> dict:
    """返回 { "ip": {"banned_at": "...", "reason": "..."}, ... }"""
    return _load_json_file(BANNED_IPS_PATH)


def _save_banned_ips(data: dict) -> None:
    _save_json_file(BANNED_IPS_PATH, data)


def _is_banned(ip: str) -> bool:
    """检查 IP 是否已封禁。"""
    banned = _load_banned_ips()
    return ip in banned


def _ban_ip(ip: str, reason: str = "") -> None:
    """封禁一个 IP。"""
    with _ban_lock:
        banned = _load_banned_ips()
        banned[ip] = {
            "banned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
        }
        _save_banned_ips(banned)
    AUDIT_LOGGER.info(json.dumps(
        {"action": "ip_banned", "ip": ip, "reason": reason},
        ensure_ascii=False, separators=(",", ":"),
    ))


def _record_404(ip: str, path: str) -> bool:
    """
    记录一次 404，返回 True 表示触发封禁。
    规则：10 秒内超过 10 次 404 即封禁。
    """
    now = time.time()
    with _404_counter_lock:
        hits = _404_counter.setdefault(ip, [])
        hits.append(now)
        # 只保留窗口内的记录
        cutoff = now - _BAN_WINDOW
        hits[:] = [t for t in hits if t > cutoff]
        if len(hits) >= _BAN_THRESHOLD:
            # 清空计数器，执行封禁
            _404_counter.pop(ip, None)
            _ban_ip(ip, reason=f"10s内{len(hits)}次404，最后路径: {path}")
            return True
    return False


# ---- IP-昵称关联 ----

def _record_ip_nickname(ip: str, nickname: str) -> None:
    """记录 IP 与昵称的关联（去重、追加）。"""
    if not ip or not nickname or ip == "unknown":
        return
    with _nick_lock:
        data = _load_json_file(IP_NICKNAMES_PATH)
        entry = data.get(ip)
        if entry is None:
            entry = {"nicknames": [], "last_seen": ""}
            data[ip] = entry
        if nickname not in entry["nicknames"]:
            entry["nicknames"].append(nickname)
        entry["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_json_file(IP_NICKNAMES_PATH, data)


# ============ 中间件：封禁检查 + 404 计数 ============

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


class BanMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = _client_ip(request)

        # 如果已封禁，直接返回 403
        if _is_banned(ip):
            return StarletteResponse(
                content="Access denied",
                status_code=403,
                media_type="text/plain",
            )

        response = await call_next(request)

        # 记录 404，判断是否需要封禁
        if response.status_code == 404:
            _record_404(ip, request.url.path)

        return response


app.add_middleware(BanMiddleware)


# /api/events 清理节流（避免 2s 轮询时频繁写锁）
_EVENTS_CLEANUP_LOCK = threading.Lock()
_EVENTS_LAST_CLEANUP_TS = 0


def _client_ip(request: Request) -> str:
    """
    Best-effort get client ip behind reverse proxies:
    - X-Forwarded-For: take the first ip
    - X-Real-IP
    - request.client.host
    """
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        # 'client, proxy1, proxy2' -> client
        first = xff.split(",")[0].strip()
        if first:
            return first
    xri = (request.headers.get("x-real-ip") or "").strip()
    if xri:
        return xri
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _audit(action: str, *, request: Request | None = None, **fields: Any) -> None:
    payload: dict[str, Any] = {"action": action}
    if request is not None:
        payload["ip"] = _client_ip(request)
        payload["method"] = request.method
        payload["path"] = request.url.path
        ua = (request.headers.get("user-agent") or "").strip()
        if ua:
            payload["ua"] = ua[:180]
    for k, v in fields.items():
        if v is None:
            continue
        payload[k] = v
    try:
        AUDIT_LOGGER.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        # Never break request flow due to logging failure
        pass


# URL 编码工具
def _urlencode(value: Any) -> str:
    return quote(str(value), safe="")


# 拼接基础 URL
def _join_base_url(base_url: str, path: str) -> str:
    if not base_url:
        return path
    base = base_url.rstrip("/")
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"


# 校验跳转地址
def _safe_next_url(next_url: str | None, *, default: str) -> str:
    if not next_url:
        return default
    next_url = str(next_url).strip()
    if next_url.startswith("/"):
        return next_url
    return default


# 追加 URL 查询参数
def _append_query_param(url: str, key: str, value: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{key}={_urlencode(value)}"


_URL_RE = re.compile(r"(?i)\bhttps?://[^\s<>()]+")


# 文本链接化（同时支持 [图片:xxx.png] 内嵌图片展示）
def _linkify(value: Any) -> Markup:
    """Convert http(s) URLs inside a string into clickable links."""
    if value is None:
        return Markup("")
    s = str(value)
    parts: list[str] = []
    last = 0
    for m in _URL_RE.finditer(s):
        parts.append(str(escape(s[last : m.start()])))
        url = m.group(0)
        parts.append(
            f'<a class="link" href="{escape(url)}" target="_blank" rel="noopener noreferrer">{escape(url)}</a>'
        )
        last = m.end()
    parts.append(str(escape(s[last:])))
    return Markup("".join(parts))


templates.env.filters["urlencode"] = _urlencode
templates.env.filters["linkify"] = _linkify


_IMAGE_RE = re.compile(r"(?i)\b([\w\-. ()\u4e00-\u9fff]+)\.(png|jpe?g|gif|webp)\b")
_IMAGE_COL_RE = re.compile(r"(?i)(图片|image|img|照片|封面)")
_DOC_SUFFIXES = {".pptx", ".docx", ".doc", ".pdf"}
_MAX_DOC_BYTES = 80 * 1024 * 1024
_MAX_DAILY_UPLOADS = 7

# Magic bytes for file type validation
_FILE_SIGNATURES = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04"],  # ZIP-based Office Open XML
    ".pptx": [b"PK\x03\x04"],  # ZIP-based Office Open XML
    ".doc": [b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"],  # OLE Compound Document
}


def _validate_file_content(content: bytes, suffix: str) -> bool:
    """Validate file content matches expected magic bytes for the file type."""
    if not content or len(content) < 8:
        return False
    signatures = _FILE_SIGNATURES.get(suffix, [])
    if not signatures:
        return False
    for sig in signatures:
        if content[:len(sig)] == sig:
            return True
    return False


def _count_today_uploads() -> int:
    """Count files uploaded today based on modification time."""
    if not DOCS_DIR.exists():
        return 0
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_ts = today_start.timestamp()
    count = 0
    for p in DOCS_DIR.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _DOC_SUFFIXES:
            continue
        if p.stat().st_mtime >= today_ts:
            count += 1
    return count


def _filter_library_files(files: list[dict[str, Any]], q: str | None) -> list[dict[str, Any]]:
    s = (q or "").strip().lower()
    if not s:
        return files
    # Support multi-keyword search: all terms must match (AND)
    terms = [t for t in re.split(r"\s+", s) if t]
    if not terms:
        return files
    out: list[dict[str, Any]] = []
    for f in files:
        name = str(f.get("name") or "").lower()
        if all(t in name for t in terms):
            out.append(f)
    return out


@dataclass(frozen=True)
class SheetData:
    name: str
    columns: list[str]
    rows: list[dict[str, Any]]
    image_column: str | None = None


@dataclass(frozen=True)
class WorkbookData:
    sheets: list[SheetData]

    @property
    def sheet_names(self) -> list[str]:
        return [s.name for s in self.sheets]

    def get(self, name: str) -> SheetData | None:
        for s in self.sheets:
            if s.name == name:
                return s
        return None


# 清理单元格值
def _clean_cell(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return v


# 判断空行
def _row_is_empty(row: dict[str, Any]) -> bool:
    return all(_clean_cell(v) is None for v in row.values())


# 兼容旧表名处理
def _safe_sheet_name(name: str) -> str:
    return name


# 识别图片字段
def _normalize_possible_image(value: Any) -> str | None:
    """
    If a cell contains an image filename or path, return the filename to serve under /images.
    """
    if not value:
        return None
    if not isinstance(value, str):
        return None

    lowered = value.replace("\\", "/")
    if "/图片/" in f"/{lowered}":
        candidate = lowered.split("/图片/")[-1].strip()
        if _IMAGE_RE.search(candidate):
            return os.path.basename(candidate)

    m = _IMAGE_RE.search(value)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2)}"


# 遍历文本字段
def _iter_texts(row: dict[str, Any]) -> Iterable[str]:
    for v in row.values():
        v = _clean_cell(v)
        if v is None:
            continue
        if isinstance(v, str):
            yield v
        else:
            yield str(v)


# 行数组转文本
def _lines_to_text(lines: Iterable[str]) -> str:
    return "\n".join(lines)


# 文本转行数组
def _text_to_lines(value: str | None) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


# 文件大小格式化
def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def _list_library_files() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not DOCS_DIR.exists():
        return entries
    for p in DOCS_DIR.iterdir():
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in _DOC_SUFFIXES:
            continue
        stat = p.stat()
        # Extract extension without dot for template icon matching
        ext = suffix.lstrip(".")
        # Create display name: truncate long names for mobile
        display_name = p.stem
        if len(display_name) > 28:
            display_name = display_name[:25] + "..."
        display_name += suffix
        entries.append(
            {
                "name": p.name,
                "display_name": display_name,
                "ext": ext,
                "size_bytes": stat.st_size,
                "size_text": _format_size(stat.st_size),
                "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "mtime": stat.st_mtime,
            }
        )
    entries.sort(key=lambda item: item["mtime"], reverse=True)
    return entries


# 读取设置并拆行
def _load_setting_lines(
    conn, *, key: str, default_lines: list[str]
) -> tuple[str, list[str], int]:
    setting = guide_db.get_setting(conn, key=key)
    if setting:
        text = setting["value"]
        version = int(setting["version"])
    else:
        text = _lines_to_text(default_lines)
        version = 0
    return text, _text_to_lines(text), version


# 获取数据库连接
def get_conn():
    """Get a thread-safe database connection."""
    conn = guide_db.get_connection(DB_PATH)
    guide_db.init_db(conn)
    return conn


# 首次导入 Excel
def bootstrap_from_xlsx_if_needed() -> None:
    """
    First run: import xlsx into sqlite for faster read/write and to support in-web editing.
    Subsequent runs: read from sqlite only.
    """
    conn = get_conn()
    try:
        if guide_db.has_any_data(conn):
            return
        if not EXCEL_PATH.exists():
            return

        xls = pd.ExcelFile(EXCEL_PATH)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            df.columns = [str(c).strip() if c is not None else "" for c in df.columns]
            df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]

            columns = [str(c) for c in df.columns]
            image_column = None
            for c in columns:
                if c and _IMAGE_COL_RE.search(c):
                    image_column = c
                    break

            rows: list[dict[str, Any]] = []
            for _, r in df.iterrows():
                row = {col: _clean_cell(r.get(col)) for col in df.columns}
                if _row_is_empty(row):
                    continue
                rows.append(row)

            guide_db.upsert_sheet(
                conn,
                name=str(sheet_name),
                columns=columns,
                image_column=image_column,
            )
            guide_db.replace_sheet_rows(conn, sheet=str(sheet_name), rows=rows)
    finally:
        conn.close()


# 加载工作簿到内存
def load_workbook() -> WorkbookData:
    """Load workbook data fresh from database (no caching for concurrent safety)."""
    bootstrap_from_xlsx_if_needed()
    conn = get_conn()
    meta = guide_db.list_sheets(conn)
    sheets: list[SheetData] = []
    for s in meta:
        db_rows = guide_db.list_rows(conn, sheet=s["name"])
        rows = [{"__id__": r.id, **r.data} for r in db_rows]
        columns = list(s["columns"])
        # UX: extend some sheets with extra optional fields without requiring a DB schema migration
        if s["name"] == "其他设备":
            for extra in ["购买链接", "推荐人"]:
                if extra not in columns:
                    columns.append(extra)
        sheets.append(
            SheetData(
                name=s["name"],
                columns=columns,
                rows=rows,
                image_column=s["image_column"],
            )
        )
    return WorkbookData(sheets=sheets)


# 筛选行数据
def _filter_rows(
    rows: list[dict[str, Any]],
    q: str | None,
    column: str | None,
    value: str | None,
) -> list[dict[str, Any]]:
    out = rows
    if q:
        q_low = q.strip().lower()
        if q_low:
            out = [
                r
                for r in out
                if any(q_low in t.lower() for t in _iter_texts(r))
            ]
    if column and value:
        col = column.strip()
        val = value.strip()
        if col and val:
            out = [
                r
                for r in out
                if _clean_cell(r.get(col)) is not None
                and val in str(_clean_cell(r.get(col)))
            ]
    return out


_CATEGORY_COL_RE = re.compile(r"(?i)(类别|分类|类型|品类|category|type)")
_NAME_COL_RE = re.compile(r"(?i)(名称|名字|设备|物品|型号|name|title)")


def _sort_key_text(v: Any) -> str:
    """Normalize a value for sorting (None last handled by caller)."""
    vv = _clean_cell(v)
    if vv is None:
        return ""
    return str(vv).strip().lower()


def _sort_rows_for_sheet(
    rows: list[dict[str, Any]], sheet_name: str, columns: list[str] | None
) -> list[dict[str, Any]]:
    """Sheet-specific ordering tweaks for better UX."""
    if sheet_name != "其他设备":
        return rows
    cols = list(columns or [])
    cat_col = next((c for c in cols if _CATEGORY_COL_RE.search(c)), None)
    if not cat_col:
        return rows
    name_col = next((c for c in cols if _NAME_COL_RE.search(c)), None)

    def key(r: dict[str, Any]):
        cat = _clean_cell(r.get(cat_col))
        name = _clean_cell(r.get(name_col)) if name_col else None
        # None categories go last; within category sort by name-ish column (if any)
        return (
            cat is None,
            _sort_key_text(cat),
            name is None,
            _sort_key_text(name),
            _sort_key_text(r.get("__id__")),
        )

    return sorted(rows, key=key)


@app.get("/guide", response_class=HTMLResponse)
# 指南页
def guide(
    request: Request,
    sheet: str | None = Query(default=None, description="Sheet name"),
    q: str | None = Query(default=None, description="Search keyword"),
    column: str | None = Query(default=None, description="Column filter"),
    value: str | None = Query(default=None, description="Column filter value"),
    view: str = Query(default="cards", pattern="^(cards|table)$"),
    uploaded: str | None = Query(default=None, description="Last uploaded filename"),
    edit: int = Query(default=0, ge=0, le=1, description="Edit mode (0/1)"),
    err: str | None = Query(default=None, description="Error message"),
):
    wb = load_workbook()
    active_sheet = sheet or (wb.sheets[0].name if wb.sheets else "")
    sd = wb.get(active_sheet)

    rows: list[dict[str, Any]] = sd.rows if sd else []
    rows = _filter_rows(rows, q=q, column=column, value=value)
    rows = _sort_rows_for_sheet(rows, active_sheet, (sd.columns if sd else None))

    conn = get_conn()

    # Batch fetch row images for display
    row_ids = [r.get("__id__") for r in rows if r.get("__id__")]
    all_row_images = guide_db.list_rows_images(conn, row_ids=row_ids) if row_ids else {}

    enriched_rows: list[dict[str, Any]] = []
    for r in rows:
        img = None
        # 仅特定工作表把图片列当作预览图（避免“新手注意事项”等表误用图片列）
        if sd and sd.image_column and active_sheet in _PREVIEW_IMAGE_SHEETS:
            img = _normalize_possible_image(r.get(sd.image_column))
        rid = r.get("__id__")
        ri = all_row_images.get(rid, []) if rid else []
        enriched_rows.append({**r, "__image__": img, "__row_images__": ri})

    try:
        shoes_text, shoes_lines, shoes_version = _load_setting_lines(
            conn, key=SETTING_SHOES_TIPS, default_lines=DEFAULT_SHOES_TIPS
        )
        ticket_text, ticket_lines, ticket_version = _load_setting_lines(
            conn, key=SETTING_TICKET_TIPS, default_lines=DEFAULT_TICKET_TIPS
        )
    finally:
        conn.close()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "深圳爬墙区新手村指南",
            "workbook": wb,
            "active_sheet": _safe_sheet_name(active_sheet),
            "sheet": sd,
            "rows": enriched_rows,
            "q": q or "",
            "column": column or "",
            "value": value or "",
            "view": view,
            "uploaded": uploaded or "",
            "images_mounted": True,
            "excel_exists": (EXCEL_PATH.exists() or DB_PATH.exists()),
            "enable_edit": True,
            "edit": bool(edit),
            "err": err or "",
            "shoes_tips_text": shoes_text,
            "shoes_tips_lines": shoes_lines,
            "shoes_tips_version": shoes_version,
            "ticket_tips_text": ticket_text,
            "ticket_tips_lines": ticket_lines,
            "ticket_tips_version": ticket_version,
            "base_path": "/guide",
        },
    )


@app.get("/healthz")
# 健康检查
def healthz():
    wb = load_workbook()
    return {
        "ok": True,
        "excel_exists": EXCEL_PATH.exists(),
        "sheets": wb.sheet_names,
        "images_dir_exists": IMAGES_DIR.exists(),
        "library_dir_exists": DOCS_DIR.exists(),
    }


@app.get("/library", response_class=HTMLResponse)
# 群图书馆
async def library_page(
    request: Request,
    q: str | None = Query(default=None, description="Keyword search"),
    err: str | None = None,
    msg: str | None = None,
):
    all_files = _list_library_files()
    files = _filter_library_files(all_files, q)
    today_uploads = _count_today_uploads()
    can_upload = today_uploads < _MAX_DAILY_UPLOADS
    return templates.TemplateResponse(
        "library.html",
        {
            "request": request,
            "title": "群图书馆",
            "excel_exists": (EXCEL_PATH.exists() or DB_PATH.exists()),
            "images_mounted": True,
            "err": err or "",
            "msg": msg or "",
            "files": files,
            "all_files_count": len(all_files),
            "max_size_mb": int(_MAX_DOC_BYTES / (1024 * 1024)),
            "allowed_types": "pptx / docx / doc / pdf",
            "today_uploads": today_uploads,
            "max_daily_uploads": _MAX_DAILY_UPLOADS,
            "can_upload": can_upload,
            "q": q or "",
        },
    )


@app.post("/library/upload")
# 上传文档
async def library_upload(request: Request, file: UploadFile = File(...)):
    # Check daily upload limit first
    today_count = _count_today_uploads()
    if today_count >= _MAX_DAILY_UPLOADS:
        _audit(
            "library_upload_blocked_daily_limit",
            request=request,
            filename=(file.filename or "").strip(),
            today_uploads=today_count,
            max_daily_uploads=_MAX_DAILY_UPLOADS,
        )
        return RedirectResponse(
            url="/library?err=今日上传已达上限，请明天再试",
            status_code=303,
        )

    original = (file.filename or "").strip()
    if not original:
        _audit("library_upload_failed_no_filename", request=request)
        return RedirectResponse(url="/library?err=缺少文件名", status_code=303)
    suffix = Path(original).suffix.lower()
    if suffix not in _DOC_SUFFIXES:
        _audit("library_upload_failed_unsupported_suffix", request=request, filename=original, suffix=suffix)
        return RedirectResponse(url="/library?err=仅支持pptx/docx/doc/pdf", status_code=303)

    content = await file.read()
    if not content:
        _audit("library_upload_failed_empty", request=request, filename=original, suffix=suffix)
        return RedirectResponse(url="/library?err=文件为空", status_code=303)
    if len(content) > _MAX_DOC_BYTES:
        _audit(
            "library_upload_failed_too_large",
            request=request,
            filename=original,
            suffix=suffix,
            size_bytes=len(content),
            max_bytes=_MAX_DOC_BYTES,
        )
        return RedirectResponse(url="/library?err=文件过大（最大80MB）", status_code=303)

    # Validate file content matches extension (magic bytes check)
    if not _validate_file_content(content, suffix):
        _audit("library_upload_failed_signature_mismatch", request=request, filename=original, suffix=suffix)
        return RedirectResponse(
            url="/library?err=文件内容与扩展名不符，请确认文件类型",
            status_code=303,
        )

    safe_stem = (
        re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", Path(original).stem)[:60]
        or "document"
    )
    # Keep filename stable (no extra random suffix). If same name exists, block upload.
    out_name = f"{safe_stem}{suffix}"
    out_path = DOCS_DIR / out_name
    # Double protection: pre-check + atomic create to avoid race overwriting.
    if out_path.exists():
        _audit("library_upload_failed_exists", request=request, filename=out_name, suffix=suffix)
        return RedirectResponse(
            url="/library?err=已存在系统文件，请勿重复上传",
            status_code=303,
        )
    try:
        with out_path.open("xb") as f:
            f.write(content)
    except FileExistsError:
        _audit("library_upload_failed_exists_race", request=request, filename=out_name, suffix=suffix)
        return RedirectResponse(
            url="/library?err=已存在系统文件，请勿重复上传",
            status_code=303,
        )
    _audit(
        "library_upload_success",
        request=request,
        filename=out_name,
        suffix=suffix,
        size_bytes=len(content),
    )
    return RedirectResponse(url="/library?msg=上传成功", status_code=303)


@app.get("/library/files/{filename}")
# 下载文档
async def library_download(request: Request, filename: str):
    if not filename or "/" in filename or "\\" in filename:
        _audit("library_download_failed_invalid_filename", request=request, filename=filename)
        raise HTTPException(status_code=400, detail="invalid filename")
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        _audit("library_download_failed_invalid_basename", request=request, filename=filename)
        raise HTTPException(status_code=400, detail="invalid filename")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _DOC_SUFFIXES:
        _audit("library_download_failed_not_allowed_suffix", request=request, filename=safe_name, suffix=suffix)
        raise HTTPException(status_code=404, detail="file not found")
    path = DOCS_DIR / safe_name
    if not path.exists():
        _audit("library_download_failed_not_found", request=request, filename=safe_name, suffix=suffix)
        raise HTTPException(status_code=404, detail="file not found")
    _audit(
        "library_download",
        request=request,
        filename=safe_name,
        suffix=suffix,
        size_bytes=path.stat().st_size,
    )
    return FileResponse(path, filename=safe_name, media_type="application/octet-stream")


@app.post("/library/delete/{filename}")
# 删除文档
async def library_delete(request: Request, filename: str):
    # Validate filename to prevent path traversal
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        _audit("library_delete_failed_invalid_filename", request=request, filename=filename)
        return RedirectResponse(url="/library?err=无效的文件名", status_code=303)
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        _audit("library_delete_failed_invalid_basename", request=request, filename=filename)
        return RedirectResponse(url="/library?err=无效的文件名", status_code=303)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _DOC_SUFFIXES:
        _audit("library_delete_failed_not_allowed_suffix", request=request, filename=safe_name, suffix=suffix)
        return RedirectResponse(url="/library?err=文件不存在", status_code=303)
    path = DOCS_DIR / safe_name
    if not path.exists():
        _audit("library_delete_failed_not_found", request=request, filename=safe_name, suffix=suffix)
        return RedirectResponse(url="/library?err=文件不存在或已被删除", status_code=303)
    try:
        size_bytes = path.stat().st_size
        path.unlink()
        _audit("library_delete_success", request=request, filename=safe_name, suffix=suffix, size_bytes=size_bytes)
        return RedirectResponse(url="/library?msg=删除成功", status_code=303)
    except Exception:
        _audit("library_delete_failed_exception", request=request, filename=safe_name, suffix=suffix)
        return RedirectResponse(url="/library?err=删除失败，请稍后重试", status_code=303)


@app.post("/guide/settings/update")
# 更新指南设置
async def update_guide_settings(request: Request):
    form = await request.form()
    shoes_text = str(form.get("shoes_tips") or "").strip()
    ticket_text = str(form.get("ticket_tips") or "").strip()
    try:
        shoes_version = int(form.get("shoes_tips_version") or 0)
    except Exception:
        shoes_version = 0
    try:
        ticket_version = int(form.get("ticket_tips_version") or 0)
    except Exception:
        ticket_version = 0

    sheet = str(form.get("sheet") or "")
    q = str(form.get("q") or "")
    column = str(form.get("column") or "")
    value = str(form.get("value") or "")
    view = str(form.get("view") or "cards")

    conn = get_conn()
    try:
        ok_shoes = guide_db.update_setting_safe(
            conn, key=SETTING_SHOES_TIPS, value=shoes_text, expected_version=shoes_version
        )
        ok_ticket = guide_db.update_setting_safe(
            conn, key=SETTING_TICKET_TIPS, value=ticket_text, expected_version=ticket_version
        )
    finally:
        conn.close()

    _audit(
        "guide_settings_update",
        request=request,
        shoes_ok=ok_shoes,
        ticket_ok=ok_ticket,
        shoes_len=len(shoes_text),
        ticket_len=len(ticket_text),
        shoes_version=shoes_version,
        ticket_version=ticket_version,
    )

    params = [
        f"sheet={_urlencode(sheet)}",
        f"q={_urlencode(q)}",
        f"column={_urlencode(column)}",
        f"value={_urlencode(value)}",
        f"view={_urlencode(view)}",
        "edit=1",
    ]
    if not (ok_shoes and ok_ticket):
        params.append(f"err={_urlencode('内容已被其他人更新，请刷新后再试')}")
    return RedirectResponse(url=f"/guide?{'&'.join(params)}", status_code=303)


# 解析时间输入
def _parse_datetime_local(value: str) -> tuple[int, str] | None:
    """Parse HTML <input type="datetime-local"> value like '2026-01-30T19:30'."""
    s = (value or "").strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M")
    except Exception:
        return None
    ts = int(time.mktime(dt.timetuple()))
    return ts, dt.strftime("%Y-%m-%d %H:%M")


# 今日起点（按日期判断过期：只要是今天及以后，都不算过期）
def _today_start_ts() -> int:
    dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return int(dt.timestamp())


_WEEKDAY_LABELS = ["一", "二", "三", "四", "五", "六", "日"]


# 计算本周起始日
def _start_of_week(d: datetime) -> datetime:
    # Monday as the first day of week
    return datetime(d.year, d.month, d.day) - timedelta(days=d.weekday())


# 生成时间段文案
def _format_period(hour: int) -> str:
    # Rough time period label for the weekly board
    if 5 <= hour < 12:
        return "上午"
    if 12 <= hour < 18:
        return "下午"
    return "晚上"


# 生成含星期的时间文本
def _start_text_with_weekday(start_ts: int, start_text: str) -> str:
    # Normalize start_text to include weekday for display and sharing
    dt = None
    try:
        dt = datetime.fromtimestamp(int(start_ts))
    except Exception:
        dt = None
    if dt is None:
        try:
            dt = datetime.strptime(start_text, "%Y-%m-%d %H:%M")
        except Exception:
            return start_text
    return f"{dt.strftime('%Y-%m-%d')}（周{_WEEKDAY_LABELS[dt.weekday()]}）{dt.strftime('%H:%M')}"


@app.get("/", response_class=HTMLResponse)
@app.get("/events", response_class=HTMLResponse)
# 约攀列表页
async def events_page(request: Request, err: str | None = None, msg: str | None = None):
    conn = get_conn()
    now_ts = int(time.time())
    today_start_ts = _today_start_ts()
    # Cleanup only events before today (date-based expiration)
    guide_db.cleanup_expired_events(conn, now_ts=today_start_ts, keep_limit=5)
    until_ts = int((datetime.now() + timedelta(days=5)).timestamp())
    # Upcoming includes today (even if time already passed)
    upcoming = guide_db.list_events_upcoming(conn, now_ts=today_start_ts, until_ts=until_ts)
    expired = guide_db.list_events_expired(conn, now_ts=today_start_ts, limit=5)
    # Add weekday display text for event lists and share text
    for e in upcoming:
        e["start_text_weekday"] = _start_text_with_weekday(e["start_ts"], e["start_text"])
    for e in expired:
        e["start_text_weekday"] = _start_text_with_weekday(e["start_ts"], e["start_text"])
    today = datetime.now()
    today_date = today.date()
    # Weekly board window: current week (Mon-Sun)
    week_start_dt = _start_of_week(today)
    week_end_dt = week_start_dt + timedelta(days=7) - timedelta(seconds=1)
    week_events = guide_db.list_events_between(
        conn,
        start_ts=int(week_start_dt.timestamp()),
        end_ts=int(week_end_dt.timestamp()),
    )

    # Group events by date for week board rendering
    week_events_by_date: dict[datetime.date, list[tuple[datetime, dict[str, Any]]]] = {}
    for e in week_events:
        event_dt = datetime.fromtimestamp(int(e["start_ts"]))
        key = event_dt.date()
        week_events_by_date.setdefault(key, []).append((event_dt, e))

    # Build week board payload: 7 days with items
    week_board: list[dict[str, Any]] = []
    for i in range(7):
        day_dt = week_start_dt + timedelta(days=i)
        day_date = day_dt.date()
        day_events = sorted(week_events_by_date.get(day_date, []), key=lambda item: item[0])
        items: list[dict[str, Any]] = []
        for event_dt, e in day_events:
            items.append(
                {
                    # Display helpers for week board
                    "period": _format_period(event_dt.hour),
                    "location": e["location"],
                    "time_text": event_dt.strftime("%H:%M"),
                    "full_time": e["start_text"],
                    # Date-based "active": today and future
                    "is_active": int(e["start_ts"]) >= today_start_ts,
                    "detail_url": f"/events/{e['id']}",
                }
            )
        day_title = f"{day_dt.month}.{day_dt.day}（周{_WEEKDAY_LABELS[day_dt.weekday()]}）"
        week_board.append(
            {
                "title": day_title,
                "items": items,
                "is_today": day_date == today_date,
            }
        )

    return templates.TemplateResponse(
        "events.html",
        {
            "request": request,
            "title": "深圳爬墙区新手村",
            "excel_exists": (EXCEL_PATH.exists() or DB_PATH.exists()),
            "images_mounted": True,
            "err": err or "",
            "msg": msg or "",
            "upcoming": upcoming,
            "expired": expired,
            "week_board": week_board,
            "now_ts": int(time.time()),
            "base_url": BASE_URL,
        },
    )


@app.get("/events/{event_id}", response_class=HTMLResponse, name="event_detail")
# 活动详情页
async def event_detail_page(
    request: Request, event_id: int, err: str | None = None, msg: str | None = None
):
    conn = get_conn()
    event = guide_db.get_event(conn, event_id=event_id)
    if not event:
        return RedirectResponse(url="/?err=活动不存在或已被删除", status_code=303)

    today_start_ts = _today_start_ts()
    is_expired = int(event["start_ts"]) < today_start_ts
    event["start_text_weekday"] = _start_text_with_weekday(
        event["start_ts"], event["start_text"]
    )
    return templates.TemplateResponse(
        "event_detail.html",
        {
            "request": request,
            "title": "约攀邀请",
            "excel_exists": (EXCEL_PATH.exists() or DB_PATH.exists()),
            "images_mounted": True,
            "err": err or "",
            "msg": msg or "",
            "event": event,
            "is_expired": is_expired,
            "base_url": BASE_URL,
        },
    )


@app.post("/events/new")
# 创建活动
async def events_new(request: Request):
    form = await request.form()
    when_raw = str(form.get("when") or "")
    location = str(form.get("location") or "").strip()
    nickname = str(form.get("nickname") or "").strip()

    parsed = _parse_datetime_local(when_raw)
    if not parsed:
        _audit("event_create_failed_bad_time", request=request, when_raw=when_raw, location=location, nickname=nickname)
        return RedirectResponse(url="/events?err=时间格式不正确", status_code=303)
    start_ts, start_text = parsed

    now_ts = int(time.time())
    until_ts = int((datetime.now() + timedelta(days=5)).timestamp())
    if start_ts < now_ts or start_ts > until_ts:
        _audit(
            "event_create_failed_time_out_of_range",
            request=request,
            start_ts=start_ts,
            start_text=start_text,
            location=location,
            nickname=nickname,
        )
        return RedirectResponse(url="/events?err=只允许预约未来五天内的活动", status_code=303)
    if not location:
        _audit("event_create_failed_no_location", request=request, start_text=start_text, nickname=nickname)
        return RedirectResponse(url="/events?err=请填写地点", status_code=303)
    if not nickname:
        _audit("event_create_failed_no_nickname", request=request, start_text=start_text, location=location)
        return RedirectResponse(url="/events?err=请填写昵称", status_code=303)

    # 记录 IP-昵称关联
    _record_ip_nickname(_client_ip(request), nickname)

    conn = get_conn()
    event_id = guide_db.insert_event(
        conn,
        start_ts=start_ts,
        start_text=start_text,
        location=location,
        nickname=nickname,
    )
    _audit(
        "event_create",
        request=request,
        event_id=event_id,
        start_ts=start_ts,
        start_text=start_text,
        location=location,
        nickname=nickname,
    )

    return RedirectResponse(url="/?msg=发起成功！", status_code=303)


@app.post("/events/{event_id}/delete")
# 删除活动
async def events_delete(request: Request, event_id: int):
    conn = get_conn()
    # fetch before delete for richer logs
    event = guide_db.get_event(conn, event_id=event_id)
    deleted = guide_db.delete_event(conn, event_id=event_id)
    if not deleted:
        _audit("event_delete_failed_not_found", request=request, event_id=event_id)
        return RedirectResponse(url="/?err=活动不存在或已被删除", status_code=303)
    _audit(
        "event_delete",
        request=request,
        event_id=event_id,
        start_text=(event or {}).get("start_text"),
        location=(event or {}).get("location"),
        host_nickname=(event or {}).get("host_nickname") or (event or {}).get("nickname"),
        participants_count=len((event or {}).get("participants") or []),
    )
    return RedirectResponse(url="/?msg=已删除", status_code=303)


@app.post("/events/{event_id}/join")
# 报名活动
async def events_join(request: Request, event_id: int):
    form = await request.form()
    nickname = str(form.get("nickname") or "").strip()
    next_url = _safe_next_url(str(form.get("next") or ""), default="/")
    if not nickname:
        _audit("event_join_failed_no_nickname", request=request, event_id=event_id)
        return RedirectResponse(
            url=_append_query_param(next_url, "err", "请填写昵称"), status_code=303
        )

    # 记录 IP-昵称关联
    _record_ip_nickname(_client_ip(request), nickname)

    conn = get_conn()
    today_start_ts = _today_start_ts()
    
    # Use atomic join operation with optimistic locking
    success, message = guide_db.join_event_atomic(
        conn, event_id=event_id, nickname=nickname, now_ts=today_start_ts
    )
    event = guide_db.get_event(conn, event_id=event_id)
    participants: list[str] = list((event or {}).get("participants") or [])

    _audit(
        "event_join",
        request=request,
        event_id=event_id,
        nickname=nickname,
        success=success,
        message=message,
        participants_count=len(participants) if event else None,
        is_participant=(nickname in participants) if event else None,
    )
    
    if success:
        return RedirectResponse(
            url=_append_query_param(next_url, "msg", message), status_code=303
        )
    return RedirectResponse(
        url=_append_query_param(next_url, "err", message), status_code=303
    )


@app.post("/events/{event_id}/leave")
# 取消报名
async def events_leave(request: Request, event_id: int):
    form = await request.form()
    nickname = str(form.get("nickname") or "").strip()
    next_url = _safe_next_url(str(form.get("next") or ""), default="/")
    if not nickname:
        _audit("event_leave_failed_no_nickname", request=request, event_id=event_id)
        return RedirectResponse(
            url=_append_query_param(next_url, "err", "缺少昵称"), status_code=303
        )
    
    # 记录 IP-昵称关联
    _record_ip_nickname(_client_ip(request), nickname)

    conn = get_conn()
    
    # Use atomic leave operation with optimistic locking
    success, message = guide_db.leave_event_atomic(conn, event_id=event_id, nickname=nickname)
    event = guide_db.get_event(conn, event_id=event_id)
    participants: list[str] = list((event or {}).get("participants") or [])

    _audit(
        "event_leave",
        request=request,
        event_id=event_id,
        nickname=nickname,
        success=success,
        message=message,
        participants_count=len(participants) if event else None,
        is_participant=(nickname in participants) if event else None,
    )
    
    if success:
        return RedirectResponse(
            url=_append_query_param(next_url, "msg", message), status_code=303
        )
    return RedirectResponse(
        url=_append_query_param(next_url, "err", message), status_code=303
    )


@app.get("/reload")
# 触发重载
def reload_excel():
    """Clear in-memory cache so changes show up without restarting the server."""
    if not ENABLE_RELOAD:
        return {"ok": False, "error": "reload disabled. set ENABLE_GUIDE_RELOAD=1"}
    return RedirectResponse(url="/", status_code=303)


@app.post("/row/new")
# 新建行
async def new_row(request: Request, sheet: str = Query(..., description="Sheet name")):
    wb = load_workbook()
    sd = wb.get(sheet)
    if not sd:
        _audit("row_create_failed_sheet_not_found", request=request, sheet=sheet)
        return JSONResponse({"ok": False, "error": "sheet not found"}, status_code=404)

    data = {c: None for c in sd.columns}
    conn = get_conn()
    row_id = guide_db.insert_row(conn, sheet=sheet, data=data)
    _audit("row_create", request=request, sheet=sheet, row_id=row_id)
    return RedirectResponse(url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}", status_code=303)


@app.get("/row/{row_id}/edit", response_class=HTMLResponse)
# 编辑行页面
async def edit_row_page(
    request: Request,
    row_id: int,
    sheet: str = Query(..., description="Sheet name"),
    err: str | None = Query(default=None),
    msg: str | None = Query(default=None),
):
    wb = load_workbook()
    sd = wb.get(sheet)
    if not sd:
        return HTMLResponse("sheet not found", status_code=404)

    conn = get_conn()
    row = guide_db.get_row(conn, row_id=row_id)
    if not row or row.sheet != sheet:
        return HTMLResponse("row not found", status_code=404)

    row_images = guide_db.list_row_images(conn, row_id=row_id)

    return templates.TemplateResponse(
        "edit_row.html",
        {
            "request": request,
            "title": f"编辑 - {sheet}",
            "sheet": sd,
            "sheet_name": sheet,
            "row_id": row_id,
            "data": row.data,
            "row_images": row_images,
            "excel_exists": (EXCEL_PATH.exists() or DB_PATH.exists()),
            "images_mounted": True,
            "enable_edit": True,
            "err": err or "",
            "msg": msg or "",
        },
    )


@app.post("/row/{row_id}/edit")
# 提交行编辑
async def edit_row_submit(request: Request, row_id: int, sheet: str = Query(..., description="Sheet name")):
    wb = load_workbook()
    sd = wb.get(sheet)
    if not sd:
        return JSONResponse({"ok": False, "error": "sheet not found"}, status_code=404)

    form = await request.form()
    data: dict[str, Any] = {}
    for c in sd.columns:
        v = form.get(c)
        v = v.strip() if isinstance(v, str) else v
        data[c] = v if v not in (None, "") else None

    conn = get_conn()
    row = guide_db.get_row(conn, row_id=row_id)
    if not row or row.sheet != sheet:
        _audit("row_update_failed_not_found", request=request, sheet=sheet, row_id=row_id)
        return JSONResponse({"ok": False, "error": "row not found"}, status_code=404)
    
    try:
        guide_db.update_row(conn, row_id=row_id, data=data)
    except guide_db.NotFoundError:
        _audit("row_update_failed_deleted", request=request, sheet=sheet, row_id=row_id)
        return JSONResponse({"ok": False, "error": "row was deleted"}, status_code=404)
    
    _audit(
        "row_update",
        request=request,
        sheet=sheet,
        row_id=row_id,
        filled_fields=sum(1 for v in data.values() if v not in (None, "")),
    )
    return RedirectResponse(url=f"/guide?sheet={_urlencode(sheet)}&edit=1", status_code=303)


@app.post("/row/{row_id}/delete")
# 删除行
async def delete_row(request: Request, row_id: int, sheet: str = Query(..., description="Sheet name")):
    conn = get_conn()
    row = guide_db.get_row(conn, row_id=row_id)
    if not row or row.sheet != sheet:
        _audit("row_delete_failed_not_found", request=request, sheet=sheet, row_id=row_id)
        return JSONResponse({"ok": False, "error": "row not found"}, status_code=404)
    guide_db.delete_row(conn, row_id=row_id)
    _audit("row_delete", request=request, sheet=sheet, row_id=row_id)
    return RedirectResponse(url=f"/guide?sheet={_urlencode(sheet)}&edit=1", status_code=303)


@app.post("/upload")
# 上传图片
async def upload_image(
    request: Request,
    sheet: str = Query(..., description="Sheet name"),
    file: UploadFile = File(...),
):
    """Upload an image into 图片/ directory."""
    wb = load_workbook()
    sd = wb.get(sheet)
    if not sd:
        _audit("image_upload_failed_sheet_not_found", request=request, sheet=sheet, filename=(file.filename or "").strip())
        return JSONResponse({"ok": False, "error": "sheet not found"}, status_code=404)
    if not sd.image_column:
        _audit("image_upload_failed_no_image_column", request=request, sheet=sheet, filename=(file.filename or "").strip())
        return JSONResponse({"ok": False, "error": "this sheet has no image column"}, status_code=400)

    original = (file.filename or "").strip()
    suffix = Path(original).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        _audit("image_upload_failed_unsupported_suffix", request=request, sheet=sheet, filename=original, suffix=suffix)
        return JSONResponse({"ok": False, "error": "unsupported file type"}, status_code=400)

    safe_stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", Path(original).stem)[:60] or "image"
    token = secrets.token_hex(4)
    out_name = f"{safe_stem}_{token}{suffix}"
    out_path = IMAGES_DIR / out_name

    content = await file.read()
    if not content:
        _audit("image_upload_failed_empty", request=request, sheet=sheet, filename=original, suffix=suffix)
        return JSONResponse({"ok": False, "error": "empty file"}, status_code=400)
    if len(content) > 10 * 1024 * 1024:
        _audit(
            "image_upload_failed_too_large",
            request=request,
            sheet=sheet,
            filename=original,
            suffix=suffix,
            size_bytes=len(content),
        )
        return JSONResponse({"ok": False, "error": "file too large (max 10MB)"}, status_code=400)

    out_path.write_bytes(content)
    _audit(
        "image_upload",
        request=request,
        sheet=sheet,
        filename=out_name,
        suffix=suffix,
        size_bytes=len(content),
    )
    return RedirectResponse(
        url=f"/guide?sheet={_urlencode(sheet)}&edit=1&uploaded={_urlencode(out_name)}",
        status_code=303,
    )



@app.post("/row/{row_id}/upload-image")
async def row_upload_image(
    request: Request,
    row_id: int,
    sheet: str = Query(..., description="Sheet name"),
    # Backward compatible: previously used to append [图片:filename] into text.
    # Now we keep it but do NOT modify any text field.
    column: str | None = Query(default=None, description="(deprecated)"),
    file: UploadFile = File(...),
):
    """Upload an image and associate it with a specific row.

    NOTE: This endpoint only uploads & associates image with the row (row_images).
    It does NOT modify any text fields.
    """
    wb = load_workbook()
    sd = wb.get(sheet)
    if not sd:
        _audit("row_image_upload_failed_sheet_not_found", request=request, sheet=sheet, row_id=row_id)
        return RedirectResponse(
            url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}&err=工作表不存在",
            status_code=303,
        )
    conn = get_conn()
    row = guide_db.get_row(conn, row_id=row_id)
    if not row or row.sheet != sheet:
        _audit("row_image_upload_failed_not_found", request=request, sheet=sheet, row_id=row_id)
        return RedirectResponse(
            url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}&err=行不存在",
            status_code=303,
        )

    original = (file.filename or "").strip()
    suffix = Path(original).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        _audit("row_image_upload_failed_suffix", request=request, sheet=sheet, row_id=row_id, filename=original)
        return RedirectResponse(
            url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}&err=仅支持png/jpg/gif/webp",
            status_code=303,
        )

    content = await file.read()
    if not content:
        return RedirectResponse(
            url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}&err=文件为空",
            status_code=303,
        )
    if len(content) > 10 * 1024 * 1024:
        return RedirectResponse(
            url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}&err=文件过大（最大10MB）",
            status_code=303,
        )

    safe_stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", Path(original).stem)[:60] or "image"
    token = secrets.token_hex(4)
    out_name = f"{safe_stem}_{token}{suffix}"
    out_path = IMAGES_DIR / out_name
    out_path.write_bytes(content)

    guide_db.insert_row_image(conn, row_id=row_id, filename=out_name, original_name=original)
    _audit("row_image_upload", request=request, sheet=sheet, row_id=row_id, filename=out_name)

    return RedirectResponse(
        url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}&msg=图片上传成功",
        status_code=303,
    )


@app.post("/row/{row_id}/delete-image/{image_id}")
async def row_delete_image(
    request: Request,
    row_id: int,
    image_id: int,
    sheet: str = Query(..., description="Sheet name"),
):
    conn = get_conn()
    img = guide_db.get_row_image(conn, image_id=image_id)
    if not img or int(img["row_id"]) != row_id:
        _audit("row_image_delete_failed", request=request, sheet=sheet, row_id=row_id, image_id=image_id)
        return RedirectResponse(
            url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}&err=图片不存在",
            status_code=303,
        )

    filename = guide_db.delete_row_image(conn, image_id=image_id)
    if filename:
        img_path = IMAGES_DIR / filename
        if img_path.exists():
            try:
                img_path.unlink()
            except Exception:
                pass
        _audit("row_image_delete", request=request, sheet=sheet, row_id=row_id, image_id=image_id, filename=filename)

    return RedirectResponse(
        url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}&msg=图片已删除",
        status_code=303,
    )


## NOTE: 回答区上传图片已统一使用 /row/{row_id}/upload-image?column=xxx


# API endpoints for AJAX operations
@app.get("/api/events")
# 约攀数据接口
def api_events():
    """Get current events data for AJAX refresh.

    NOTE: keep this endpoint non-blocking for concurrent visitors:
    - defined as sync def so FastAPI runs it in threadpool
    - avoid high-frequency write transactions; cleanup runs at low frequency
    """
    conn = get_conn()
    now_ts = int(time.time())
    today_start_ts = _today_start_ts()

    # Low-frequency cleanup to avoid write lock contention under 2s polling
    global _EVENTS_LAST_CLEANUP_TS
    if now_ts - _EVENTS_LAST_CLEANUP_TS > 120:
        if _EVENTS_CLEANUP_LOCK.acquire(blocking=False):
            try:
                if now_ts - _EVENTS_LAST_CLEANUP_TS > 120:
                    # Cleanup only events before today (date-based expiration)
                    guide_db.cleanup_expired_events(conn, now_ts=today_start_ts, keep_limit=5)
                    _EVENTS_LAST_CLEANUP_TS = now_ts
            finally:
                _EVENTS_CLEANUP_LOCK.release()

    until_ts = int((datetime.now() + timedelta(days=5)).timestamp())
    upcoming = guide_db.list_events_upcoming(conn, now_ts=today_start_ts, until_ts=until_ts)
    expired = guide_db.list_events_expired(conn, now_ts=today_start_ts, limit=5)
    # Add weekday display for client render consistency
    for e in upcoming:
        e["start_text_weekday"] = _start_text_with_weekday(e["start_ts"], e["start_text"])
    for e in expired:
        e["start_text_weekday"] = _start_text_with_weekday(e["start_ts"], e["start_text"])
    return {"upcoming": upcoming, "expired": expired, "now_ts": now_ts}


@app.get("/api/week-board")
# 村历数据接口（供前端 2s 轮询）
def api_week_board():
    conn = get_conn()
    today = datetime.now()
    today_date = today.date()
    today_start_ts = _today_start_ts()

    week_start_dt = _start_of_week(today)
    week_end_dt = week_start_dt + timedelta(days=7) - timedelta(seconds=1)
    week_events = guide_db.list_events_between(
        conn,
        start_ts=int(week_start_dt.timestamp()),
        end_ts=int(week_end_dt.timestamp()),
    )

    week_events_by_date: dict = {}
    for e in week_events:
        event_dt = datetime.fromtimestamp(int(e["start_ts"]))
        key = event_dt.date()
        week_events_by_date.setdefault(key, []).append((event_dt, e))

    week_board: list[dict[str, Any]] = []
    for i in range(7):
        day_dt = week_start_dt + timedelta(days=i)
        day_date = day_dt.date()
        day_events = sorted(week_events_by_date.get(day_date, []), key=lambda item: item[0])
        items: list[dict[str, Any]] = []
        for event_dt, e in day_events:
            items.append({
                "period": _format_period(event_dt.hour),
                "location": e["location"],
                "time_text": event_dt.strftime("%H:%M"),
                "full_time": e["start_text"],
                "is_active": int(e["start_ts"]) >= today_start_ts,
                "detail_url": f"/events/{e['id']}",
            })
        day_title = f"{day_dt.month}.{day_dt.day}（周{_WEEKDAY_LABELS[day_dt.weekday()]}）"
        week_board.append({
            "title": day_title,
            "items": items,
            "is_today": day_date == today_date,
        })

    return {"week_board": week_board}


if __name__ == "__main__":
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

    log_config = copy.deepcopy(UVICORN_LOGGING_CONFIG)
    log_config["formatters"]["default"]["fmt"] = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    log_config["formatters"]["access"]["fmt"] = (
        '%(asctime)s | %(levelname)s | %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=log_config)

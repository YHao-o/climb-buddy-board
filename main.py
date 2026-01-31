from __future__ import annotations

import copy
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, File, Query, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from urllib.parse import quote

import db as guide_db


ROOT = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / "深圳市爬墙区新手村指南.xlsx"
IMAGES_DIR = ROOT / "图片"
ASSETS_DIR = ROOT / "static"
TEMPLATES_DIR = ROOT / "templates"
DB_PATH = ROOT / "data.db"
# 站点基础地址（用于生成邀请/复制链接）
BASE_URL ="http://120.79.176.134:8000"


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

# Static assets (css/js)
if ASSETS_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(ASSETS_DIR)), name="static")

# User-provided images
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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


# 文本链接化
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
        sheets.append(
            SheetData(
                name=s["name"],
                columns=list(s["columns"]),
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

    enriched_rows: list[dict[str, Any]] = []
    for r in rows:
        img = None
        if sd and sd.image_column:
            img = _normalize_possible_image(r.get(sd.image_column))
        enriched_rows.append({**r, "__image__": img})

    conn = get_conn()
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
            "excel_exists": EXCEL_PATH.exists(),
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
    }


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
    guide_db.cleanup_expired_events(conn, now_ts=now_ts, keep_limit=5)
    until_ts = int((datetime.now() + timedelta(days=5)).timestamp())
    upcoming = guide_db.list_events_upcoming(conn, now_ts=now_ts, until_ts=until_ts)
    expired = guide_db.list_events_expired(conn, now_ts=now_ts, limit=5)
    # Add weekday display text for event lists and share text
    for e in upcoming:
        e["start_text_weekday"] = _start_text_with_weekday(e["start_ts"], e["start_text"])
    for e in expired:
        e["start_text_weekday"] = _start_text_with_weekday(e["start_ts"], e["start_text"])
    today = datetime.now()
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
                    "is_active": int(e["start_ts"]) >= now_ts,
                    "detail_url": f"/events/{e['id']}",
                }
            )
        day_title = f"{day_dt.month}.{day_dt.day}（周{_WEEKDAY_LABELS[day_dt.weekday()]}）"
        week_board.append({"title": day_title, "items": items})

    return templates.TemplateResponse(
        "events.html",
        {
            "request": request,
            "title": "深圳爬墙区新手村",
            "excel_exists": EXCEL_PATH.exists(),
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

    now_ts = int(time.time())
    is_expired = int(event["start_ts"]) < now_ts
    event["start_text_weekday"] = _start_text_with_weekday(
        event["start_ts"], event["start_text"]
    )
    return templates.TemplateResponse(
        "event_detail.html",
        {
            "request": request,
            "title": "约攀邀请",
            "excel_exists": EXCEL_PATH.exists(),
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
        return RedirectResponse(url="/events?err=时间格式不正确", status_code=303)
    start_ts, start_text = parsed

    now_ts = int(time.time())
    until_ts = int((datetime.now() + timedelta(days=5)).timestamp())
    if start_ts < now_ts or start_ts > until_ts:
        return RedirectResponse(url="/events?err=只允许预约未来五天内的活动", status_code=303)
    if not location:
        return RedirectResponse(url="/events?err=请填写地点", status_code=303)
    if not nickname:
        return RedirectResponse(url="/events?err=请填写昵称", status_code=303)

    conn = get_conn()
    guide_db.insert_event(
        conn,
        start_ts=start_ts,
        start_text=start_text,
        location=location,
        nickname=nickname,
    )

    return RedirectResponse(url="/?msg=发起成功！", status_code=303)


@app.post("/events/{event_id}/delete")
# 删除活动
async def events_delete(event_id: int):
    conn = get_conn()
    deleted = guide_db.delete_event(conn, event_id=event_id)
    if not deleted:
        return RedirectResponse(url="/?err=活动不存在或已被删除", status_code=303)
    return RedirectResponse(url="/?msg=已删除", status_code=303)


@app.post("/events/{event_id}/join")
# 报名活动
async def events_join(request: Request, event_id: int):
    form = await request.form()
    nickname = str(form.get("nickname") or "").strip()
    next_url = _safe_next_url(str(form.get("next") or ""), default="/")
    if not nickname:
        return RedirectResponse(
            url=_append_query_param(next_url, "err", "请填写昵称"), status_code=303
        )

    conn = get_conn()
    now_ts = int(time.time())
    
    # Use atomic join operation with optimistic locking
    success, message = guide_db.join_event_atomic(conn, event_id=event_id, nickname=nickname, now_ts=now_ts)
    
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
        return RedirectResponse(
            url=_append_query_param(next_url, "err", "缺少昵称"), status_code=303
        )
    
    conn = get_conn()
    
    # Use atomic leave operation with optimistic locking
    success, message = guide_db.leave_event_atomic(conn, event_id=event_id, nickname=nickname)
    
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
async def new_row(sheet: str = Query(..., description="Sheet name")):
    wb = load_workbook()
    sd = wb.get(sheet)
    if not sd:
        return JSONResponse({"ok": False, "error": "sheet not found"}, status_code=404)

    data = {c: None for c in sd.columns}
    conn = get_conn()
    row_id = guide_db.insert_row(conn, sheet=sheet, data=data)
    return RedirectResponse(url=f"/row/{row_id}/edit?sheet={_urlencode(sheet)}", status_code=303)


@app.get("/row/{row_id}/edit", response_class=HTMLResponse)
# 编辑行页面
async def edit_row_page(request: Request, row_id: int, sheet: str = Query(..., description="Sheet name")):
    wb = load_workbook()
    sd = wb.get(sheet)
    if not sd:
        return HTMLResponse("sheet not found", status_code=404)

    conn = get_conn()
    row = guide_db.get_row(conn, row_id=row_id)
    if not row or row.sheet != sheet:
        return HTMLResponse("row not found", status_code=404)

    return templates.TemplateResponse(
        "edit_row.html",
        {
            "request": request,
            "title": f"编辑 - {sheet}",
            "sheet": sd,
            "sheet_name": sheet,
            "row_id": row_id,
            "data": row.data,
            "excel_exists": EXCEL_PATH.exists(),
            "images_mounted": True,
            "enable_edit": True,
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
        return JSONResponse({"ok": False, "error": "row not found"}, status_code=404)
    
    try:
        guide_db.update_row(conn, row_id=row_id, data=data)
    except guide_db.NotFoundError:
        return JSONResponse({"ok": False, "error": "row was deleted"}, status_code=404)
    
    return RedirectResponse(url=f"/?sheet={_urlencode(sheet)}&edit=1", status_code=303)


@app.post("/row/{row_id}/delete")
# 删除行
async def delete_row(row_id: int, sheet: str = Query(..., description="Sheet name")):
    conn = get_conn()
    row = guide_db.get_row(conn, row_id=row_id)
    if not row or row.sheet != sheet:
        return JSONResponse({"ok": False, "error": "row not found"}, status_code=404)
    guide_db.delete_row(conn, row_id=row_id)
    return RedirectResponse(url=f"/?sheet={_urlencode(sheet)}&edit=1", status_code=303)


@app.post("/upload")
# 上传图片
async def upload_image(
    sheet: str = Query(..., description="Sheet name"),
    file: UploadFile = File(...),
):
    """Upload an image into 图片/ directory."""
    wb = load_workbook()
    sd = wb.get(sheet)
    if not sd:
        return JSONResponse({"ok": False, "error": "sheet not found"}, status_code=404)
    if not sd.image_column:
        return JSONResponse({"ok": False, "error": "this sheet has no image column"}, status_code=400)

    original = (file.filename or "").strip()
    suffix = Path(original).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return JSONResponse({"ok": False, "error": "unsupported file type"}, status_code=400)

    safe_stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", Path(original).stem)[:60] or "image"
    token = secrets.token_hex(4)
    out_name = f"{safe_stem}_{token}{suffix}"
    out_path = IMAGES_DIR / out_name

    content = await file.read()
    if not content:
        return JSONResponse({"ok": False, "error": "empty file"}, status_code=400)
    if len(content) > 10 * 1024 * 1024:
        return JSONResponse({"ok": False, "error": "file too large (max 10MB)"}, status_code=400)

    out_path.write_bytes(content)
    return RedirectResponse(
        url=f"/?sheet={_urlencode(sheet)}&uploaded={_urlencode(out_name)}",
        status_code=303,
    )


# API endpoints for AJAX operations
@app.get("/api/events")
# 约攀数据接口
async def api_events():
    """Get current events data for AJAX refresh."""
    conn = get_conn()
    now_ts = int(time.time())
    guide_db.cleanup_expired_events(conn, now_ts=now_ts, keep_limit=5)
    until_ts = int((datetime.now() + timedelta(days=5)).timestamp())
    upcoming = guide_db.list_events_upcoming(conn, now_ts=now_ts, until_ts=until_ts)
    expired = guide_db.list_events_expired(conn, now_ts=now_ts, limit=5)
    return {"upcoming": upcoming, "expired": expired, "now_ts": now_ts}


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

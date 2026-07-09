from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

WEEKDAY_LABELS = ["一", "二", "三", "四", "五", "六", "日"]
PERIOD_KEYS = ["morning", "afternoon", "evening"]
PERIOD_LABELS = {"morning": "上午", "afternoon": "下午", "evening": "晚上"}


def classify_period(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def start_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _event_dt(e: dict[str, Any]) -> datetime:
    return datetime.fromtimestamp(int(e["start_ts"]))


def build_week(events: list[dict], *, today: date, now_ts: int) -> list[dict]:
    week_start = start_of_week(today)
    by_date: dict[date, list[dict]] = {}
    for e in events:
        dt = _event_dt(e)
        by_date.setdefault(dt.date(), []).append(e)

    out: list[dict] = []
    for i in range(7):
        day = week_start + timedelta(days=i)
        day_events = sorted(by_date.get(day, []), key=lambda e: int(e["start_ts"]))
        items = []
        for e in day_events:
            dt = _event_dt(e)
            period = classify_period(dt.hour)
            items.append({
                "period": period,
                "period_label": PERIOD_LABELS[period],
                "location": e["location"],
                "time_text": dt.strftime("%H:%M"),
                "full_time": e["start_text"],
                "is_active": int(e["start_ts"]) >= now_ts,
                "detail_url": f"/events/{e['id']}",
            })
        out.append({
            "title": f"{day.month}.{day.day}（周{WEEKDAY_LABELS[day.weekday()]}）",
            "is_today": day == today,
            "items": items,
        })
    return out


MONTH_MAX_LINES = 2
MONTH_MAX_DOTS = 3


def _month_anchor(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def build_month(events: list[dict], *, year: int, month: int, today: date) -> dict:
    by_date: dict[date, list[dict]] = {}
    for e in events:
        dt = _event_dt(e)
        by_date.setdefault(dt.date(), []).append(e)

    first = date(year, month, 1)
    grid_start = start_of_week(first)  # 周一

    prev_year, prev_month = (year - 1, 12) if month == 1 else (year, month - 1)
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)

    weeks: list[list[dict]] = []
    for w in range(6):
        row: list[dict] = []
        for d in range(7):
            day = grid_start + timedelta(days=w * 7 + d)
            day_events = sorted(by_date.get(day, []), key=lambda e: int(e["start_ts"]))
            visible = []
            for e in day_events[:MONTH_MAX_LINES]:
                dt = _event_dt(e)
                visible.append({
                    "time_text": dt.strftime("%H:%M"),
                    "location": e["location"],
                    "period": classify_period(dt.hour),
                    "detail_url": f"/events/{e['id']}",
                })
            dots = [classify_period(_event_dt(e).hour) for e in day_events[:MONTH_MAX_DOTS]]
            total = len(day_events)
            row.append({
                "date": day.strftime("%Y-%m-%d"),
                "day_num": day.day,
                "in_month": day.month == month and day.year == year,
                "is_today": day == today,
                "visible_items": visible,
                "dots": dots,
                "more_count": max(0, total - MONTH_MAX_LINES),
                "total": total,
            })
        weeks.append(row)

    return {
        "year": year,
        "month": month,
        "title": f"{year}年{month}月",
        "prev_anchor": _month_anchor(prev_year, prev_month),
        "next_anchor": _month_anchor(next_year, next_month),
        "is_current_month": (today.year == year and today.month == month),
        "weekday_headers": list(WEEKDAY_LABELS),
        "weeks": weeks,
    }


def build_list(events: list[dict], *, today: date) -> list[dict]:
    by_date: dict[date, list[dict]] = {}
    for e in events:
        dt = _event_dt(e)
        by_date.setdefault(dt.date(), []).append(e)

    groups: list[dict] = []
    for day in sorted(by_date.keys()):
        day_events = sorted(by_date[day], key=lambda e: int(e["start_ts"]))
        enriched = []
        for e in day_events:
            item = dict(e)
            item["time_text"] = _event_dt(e).strftime("%H:%M")
            enriched.append(item)
        groups.append({
            "date": day.strftime("%Y-%m-%d"),
            "title": f"{day.month}.{day.day}（周{WEEKDAY_LABELS[day.weekday()]}）",
            "is_today": day == today,
            "is_past": day < today,
            "events": enriched,
        })
    return groups


_WEEKDAY_FULL = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def build_heatmap(events: list[dict]) -> dict:
    counts: dict[tuple[int, str], int] = {}
    for e in events:
        dt = _event_dt(e)
        key = (dt.weekday(), classify_period(dt.hour))
        counts[key] = counts.get(key, 0) + 1

    max_count = max(counts.values()) if counts else 0

    rows = []
    for wi in range(7):
        cells = []
        for period in PERIOD_KEYS:
            c = counts.get((wi, period), 0)
            level = 0 if c == 0 else math.ceil(c / max_count * 4)
            cells.append({"period": period, "count": c, "level": level})
        rows.append({"weekday_label": _WEEKDAY_FULL[wi], "weekday_index": wi, "cells": cells})

    if max_count == 0:
        summary = "暂无活动数据"
    else:
        best_wi, best_period = max(counts.items(), key=lambda kv: kv[1])[0]
        summary = f"{_WEEKDAY_FULL[best_wi]}{PERIOD_LABELS[best_period]}最热门"

    return {"rows": rows, "max_count": max_count, "summary": summary}


_FEED_ICONS = {"create": "🟢", "join": "🙋", "leave": "🔴", "delete": "🗑️"}
_FEED_VERBS = {"create": "发起了", "join": "报名了", "leave": "取消了", "delete": "删除了"}


def _relative_time(delta_seconds: int) -> str:
    if delta_seconds < 60:
        return "刚刚"
    if delta_seconds < 3600:
        return f"{delta_seconds // 60}分钟前"
    if delta_seconds < 86400:
        return f"{delta_seconds // 3600}小时前"
    return f"{delta_seconds // 86400}天前"


def _short_when(start_text: str) -> str:
    try:
        dt = datetime.strptime(start_text, "%Y-%m-%d %H:%M")
        return f"{dt.month}.{dt.day}"
    except (ValueError, TypeError):
        return start_text


def build_feed(activities: list[dict], *, now_ts: int) -> list[dict]:
    out = []
    for a in activities:
        action = a.get("action", "")
        delta = max(0, now_ts - int(a.get("ts", now_ts)))
        out.append({
            "action": action,
            "icon": _FEED_ICONS.get(action, "•"),
            "verb": _FEED_VERBS.get(action, action),
            "actor": a.get("actor", ""),
            "when": _short_when(a.get("start_text", "")),
            "location": a.get("location", ""),
            "relative_time": _relative_time(delta),
        })
    return out





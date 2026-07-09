"""连锁品牌分组（岩馆按连锁品牌归类）。

数据存放在 settings 表：
  - chain_category_order      -> JSON 数组，分组显示顺序（分组 id 列表）
  - chain_category:<id>       -> JSON 对象，单个分组的定义
        {name, emoji, keywords:[...], manual_gyms:[...], intro,
         bring_newbie, branch_newbie}

岩馆归入哪个分组的优先级：
  1) manual_gyms 显式指定（按归一化后的名称匹配，优先级最高）
  2) 关键词 keywords 或 分组名称 命中岩馆名称
  3) 都不命中 -> 未分组
"""
from __future__ import annotations

import json
import secrets
from typing import Any

import db as guide_db

GYM_SHEET = "岩馆"

SETTING_PREFIX = "chain_category:"
SETTING_ORDER = "chain_category_order"
SETTING_SEEDED = "chain_category_seeded"

# 岩馆名称所在的列（用于匹配/显示），按优先级
_NAME_KEYS = ("名称", "具体位置")

# 未分组桶的固定 id
UNGROUPED_ID = "__ungrouped__"


# ---------------------------------------------------------------- 工具

def _norm(text: Any) -> str:
    """归一化岩馆名称：小写、去空白、全角括号转半角、去掉常见后缀字符，便于宽松比较。"""
    s = str(text or "").lower()
    table = {
        "（": "(", "）": ")", "【": "(", "】": ")", "［": "(", "］": ")",
        " ": "", "　": "", "(": "", ")": "", "·": "", "．": "", ".": "",
        "店": "", "馆": "",
    }
    return "".join(table.get(ch, ch) for ch in s)


def _gym_names(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k in _NAME_KEYS:
        v = row.get(k)
        if v is not None and str(v).strip():
            out.append(str(v).strip())
    return out


def gym_identifier(row: dict[str, Any]) -> str:
    """用于“归入品牌”表单的稳定标识（优先名称，其次具体位置）。"""
    names = _gym_names(row)
    return names[0] if names else ""


def _row_matches_manual(row: dict[str, Any], manual_gyms: list[str]) -> bool:
    norm_names = [_norm(n) for n in _gym_names(row)]
    for m in manual_gyms:
        nm = _norm(m)
        if not nm:
            continue
        for nn in norm_names:
            if nn and (nn == nm or nn in nm or nm in nn):
                return True
    return False


def _row_matches_keywords(row: dict[str, Any], category: dict[str, Any]) -> bool:
    haystack = " ".join(_gym_names(row)).lower()
    if not haystack:
        return False
    needles = [k.lower().strip() for k in category.get("keywords", []) if str(k).strip()]
    # 分组名称本身也作为隐式关键词（如 CUBE / 蓝天攀岩 / Upper）
    name = str(category.get("name") or "").lower().strip()
    if len(name) >= 2:
        needles.append(name)
    return any(n and n in haystack for n in needles)


# ---------------------------------------------------------------- 读取

def _normalize_category(cat_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": cat_id,
        "name": str(raw.get("name") or "未命名分组"),
        "emoji": str(raw.get("emoji") or "🧗"),
        "keywords": [str(k) for k in (raw.get("keywords") or []) if str(k).strip()],
        "manual_gyms": [str(g) for g in (raw.get("manual_gyms") or []) if str(g).strip()],
        "intro": str(raw.get("intro") or ""),
        "bring_newbie": str(raw.get("bring_newbie") or ""),
        "branch_newbie": raw.get("branch_newbie") or {},
    }


def load_categories(conn) -> list[dict[str, Any]]:
    """按显示顺序返回所有分组定义。"""
    cats: dict[str, dict[str, Any]] = {}
    for row in guide_db.list_settings_by_prefix(conn, prefix=SETTING_PREFIX):
        cat_id = row["key"][len(SETTING_PREFIX):]
        try:
            raw = json.loads(row["value"])
        except Exception:
            continue
        if isinstance(raw, dict):
            cats[cat_id] = _normalize_category(cat_id, raw)

    order = _load_order(conn)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cat_id in order:
        if cat_id in cats:
            ordered.append(cats[cat_id])
            seen.add(cat_id)
    # 顺序里没有的分组追加在后面（按名称稳定排序）
    for cat_id in sorted(cats):
        if cat_id not in seen:
            ordered.append(cats[cat_id])
    return ordered


def _load_order(conn) -> list[str]:
    s = guide_db.get_setting(conn, key=SETTING_ORDER)
    if not s:
        return []
    try:
        val = json.loads(s["value"])
        return [str(x) for x in val] if isinstance(val, list) else []
    except Exception:
        return []


def group_rows(
    rows: list[dict[str, Any]], categories: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """把岩馆行按分组归类，返回有序的分组列表（含未分组桶）。

    每个返回项：{id, name, emoji, intro, bring_newbie, keywords, count, rows}
    其中每行会被打上 __group_id__ 标记，方便模板里显示当前归属。
    """
    buckets: dict[str, list[dict[str, Any]]] = {c["id"]: [] for c in categories}
    ungrouped: list[dict[str, Any]] = []

    for row in rows:
        chosen: str | None = None
        # 1) 手动指定优先
        for c in categories:
            if _row_matches_manual(row, c["manual_gyms"]):
                chosen = c["id"]
                break
        # 2) 关键词/名称匹配
        if chosen is None:
            for c in categories:
                if _row_matches_keywords(row, c):
                    chosen = c["id"]
                    break
        row["__group_id__"] = chosen or ""
        if chosen is None:
            ungrouped.append(row)
        else:
            buckets[chosen].append(row)

    result: list[dict[str, Any]] = []
    for c in categories:
        result.append({
            "id": c["id"],
            "name": c["name"],
            "emoji": c["emoji"],
            "intro": c["intro"],
            "bring_newbie": c["bring_newbie"],
            "keywords": c["keywords"],
            "rows": buckets[c["id"]],
            "count": len(buckets[c["id"]]),
            "is_ungrouped": False,
        })
    if ungrouped:
        result.append({
            "id": UNGROUPED_ID,
            "name": "未分组",
            "emoji": "📍",
            "intro": "",
            "bring_newbie": "",
            "keywords": [],
            "rows": ungrouped,
            "count": len(ungrouped),
            "is_ungrouped": True,
        })
    return result


# ---------------------------------------------------------------- 写入

def _save_order(conn, order: list[str]) -> None:
    guide_db.put_setting(conn, key=SETTING_ORDER, value=json.dumps(order, ensure_ascii=False))


def _save_category(conn, cat_id: str, data: dict[str, Any]) -> None:
    guide_db.put_setting(
        conn,
        key=SETTING_PREFIX + cat_id,
        value=json.dumps(data, ensure_ascii=False),
    )


def create_category(conn, *, name: str, emoji: str, keywords: list[str]) -> str:
    cat_id = "cat_" + secrets.token_hex(6)
    data = {
        "name": name.strip() or "未命名分组",
        "emoji": emoji.strip() or "🧗",
        "keywords": [k.strip() for k in keywords if k.strip()],
        "manual_gyms": [],
        "intro": "",
        "bring_newbie": "未知",
        "branch_newbie": {},
    }
    _save_category(conn, cat_id, data)
    order = _load_order(conn)
    order.append(cat_id)
    _save_order(conn, order)
    return cat_id


def update_category(
    conn,
    *,
    cat_id: str,
    name: str,
    emoji: str,
    keywords: list[str],
    intro: str,
    bring_newbie: str,
) -> bool:
    s = guide_db.get_setting(conn, key=SETTING_PREFIX + cat_id)
    if not s:
        return False
    try:
        data = json.loads(s["value"])
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    data["name"] = name.strip() or data.get("name") or "未命名分组"
    data["emoji"] = emoji.strip() or data.get("emoji") or "🧗"
    data["keywords"] = [k.strip() for k in keywords if k.strip()]
    data["intro"] = intro.strip()
    data["bring_newbie"] = bring_newbie.strip()
    data.setdefault("manual_gyms", [])
    data.setdefault("branch_newbie", {})
    _save_category(conn, cat_id, data)
    return True


def delete_category(conn, *, cat_id: str) -> bool:
    ok = guide_db.delete_setting(conn, key=SETTING_PREFIX + cat_id)
    order = [c for c in _load_order(conn) if c != cat_id]
    _save_order(conn, order)
    return ok


def move_category(conn, *, cat_id: str, direction: str) -> bool:
    order = _load_order(conn)
    if cat_id not in order:
        # 用当前加载顺序兜底重建 order
        order = [c["id"] for c in load_categories(conn)]
    if cat_id not in order:
        return False
    i = order.index(cat_id)
    j = i - 1 if direction == "up" else i + 1
    if j < 0 or j >= len(order):
        return False
    order[i], order[j] = order[j], order[i]
    _save_order(conn, order)
    return True


def assign_gym(conn, *, gym: str, cat_id: str) -> bool:
    """把某岩馆显式归入某分组；cat_id 为空表示从所有分组的 manual_gyms 中移除。

    会先把该岩馆从其它分组的 manual_gyms 里摘掉，避免重复归属。
    """
    gym = gym.strip()
    if not gym:
        return False
    gym_norm = _norm(gym)
    changed = False
    for cat in load_categories(conn):
        cid = cat["id"]
        manual = list(cat["manual_gyms"])
        new_manual = [m for m in manual if _norm(m) != gym_norm]
        if cid == cat_id and not any(_norm(m) == gym_norm for m in new_manual):
            new_manual.append(gym)
        if new_manual != manual:
            s = guide_db.get_setting(conn, key=SETTING_PREFIX + cid)
            if not s:
                continue
            try:
                data = json.loads(s["value"])
            except Exception:
                data = {}
            data["manual_gyms"] = new_manual
            _save_category(conn, cid, data)
            changed = True
    return changed

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator


@dataclass(frozen=True)
class DbRow:
    id: int
    sheet: str
    data: dict[str, Any]


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class ConcurrencyError(DatabaseError):
    """Raised when a concurrent modification conflict occurs."""
    pass


class NotFoundError(DatabaseError):
    """Raised when a requested resource is not found."""
    pass


# Thread-local storage for connections
_local = threading.local()


# 获取数据库连接
def get_connection(db_path: Path) -> sqlite3.Connection:
    """Get a thread-local database connection with WAL mode enabled."""
    key = str(db_path)
    if not hasattr(_local, 'connections'):
        _local.connections = {}
    
    conn = _local.connections.get(key)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            conn = None
            _local.connections.pop(key, None)

    if conn is None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
        _local.connections[key] = conn
    
    return _local.connections[key]


@contextmanager
# 事务封装
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Cursor, None, None]:
    """Context manager for database transactions with automatic rollback on error."""
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")  # Acquire write lock immediately
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


# 初始化数据库结构
def init_db(conn: sqlite3.Connection) -> None:
    with transaction(conn) as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sheets (
              name TEXT PRIMARY KEY,
              columns_json TEXT NOT NULL,
              image_column TEXT NULL,
              updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rows (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              sheet_name TEXT NOT NULL,
              data_json TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              updated_at TEXT NOT NULL DEFAULT (datetime('now')),
              version INTEGER NOT NULL DEFAULT 1,
              FOREIGN KEY(sheet_name) REFERENCES sheets(name) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rows_sheet ON rows(sheet_name)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS row_images (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              row_id INTEGER NOT NULL,
              filename TEXT NOT NULL,
              original_name TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY(row_id) REFERENCES rows(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_row_images_row ON row_images(row_id)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              start_ts INTEGER NOT NULL,
              start_text TEXT NOT NULL,
              location TEXT NOT NULL,
              nickname TEXT NOT NULL,
              host_nickname TEXT NOT NULL DEFAULT '',
              participants_json TEXT NOT NULL DEFAULT '[]',
              version INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_start_ts ON events(start_ts)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              version INTEGER NOT NULL DEFAULT 1,
              updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        
        # Migration for older dbs
        cur.execute("PRAGMA table_info(events)")
        cols = {r[1] for r in cur.fetchall()}
        if "host_nickname" not in cols:
            cur.execute("ALTER TABLE events ADD COLUMN host_nickname TEXT NOT NULL DEFAULT ''")
        if "participants_json" not in cols:
            cur.execute("ALTER TABLE events ADD COLUMN participants_json TEXT NOT NULL DEFAULT '[]'")
        if "version" not in cols:
            cur.execute("ALTER TABLE events ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
        
        # Check rows table for version column
        cur.execute("PRAGMA table_info(rows)")
        row_cols = {r[1] for r in cur.fetchall()}
        if "version" not in row_cols:
            cur.execute("ALTER TABLE rows ADD COLUMN version INTEGER NOT NULL DEFAULT 1")


# 判断是否有数据
def has_any_data(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sheets LIMIT 1")
    return cur.fetchone() is not None


# 新增或更新表
def upsert_sheet(
    conn: sqlite3.Connection,
    *,
    name: str,
    columns: list[str],
    image_column: str | None,
) -> None:
    with transaction(conn) as cur:
        cur.execute(
            """
            INSERT INTO sheets(name, columns_json, image_column, updated_at)
            VALUES(?, ?, ?, datetime('now'))
            ON CONFLICT(name) DO UPDATE SET
              columns_json=excluded.columns_json,
              image_column=excluded.image_column,
              updated_at=datetime('now')
            """,
            (name, json.dumps(columns, ensure_ascii=False), image_column),
        )


# 全量替换表数据
def replace_sheet_rows(conn: sqlite3.Connection, *, sheet: str, rows: list[dict[str, Any]]) -> None:
    with transaction(conn) as cur:
        cur.execute("DELETE FROM rows WHERE sheet_name=?", (sheet,))
        cur.executemany(
            "INSERT INTO rows(sheet_name, data_json) VALUES(?, ?)",
            [(sheet, json.dumps(r, ensure_ascii=False)) for r in rows],
        )


# 列出表信息
def list_sheets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT name, columns_json, image_column FROM sheets ORDER BY name")
    out: list[dict[str, Any]] = []
    for r in cur.fetchall():
        out.append(
            {
                "name": r["name"],
                "columns": json.loads(r["columns_json"]),
                "image_column": r["image_column"],
            }
        )
    return out


# 获取单个表
def get_sheet(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    cur = conn.cursor()
    cur.execute("SELECT name, columns_json, image_column FROM sheets WHERE name=?", (name,))
    r = cur.fetchone()
    if not r:
        return None
    return {
        "name": r["name"],
        "columns": json.loads(r["columns_json"]),
        "image_column": r["image_column"],
    }


# 列出表行
def list_rows(conn: sqlite3.Connection, *, sheet: str) -> list[DbRow]:
    cur = conn.cursor()
    cur.execute("SELECT id, sheet_name, data_json FROM rows WHERE sheet_name=? ORDER BY id", (sheet,))
    rows: list[DbRow] = []
    for r in cur.fetchall():
        rows.append(DbRow(id=int(r["id"]), sheet=str(r["sheet_name"]), data=json.loads(r["data_json"])))
    return rows


# 获取单行
def get_row(conn: sqlite3.Connection, *, row_id: int) -> DbRow | None:
    cur = conn.cursor()
    cur.execute("SELECT id, sheet_name, data_json FROM rows WHERE id=?", (row_id,))
    r = cur.fetchone()
    if not r:
        return None
    return DbRow(id=int(r["id"]), sheet=str(r["sheet_name"]), data=json.loads(r["data_json"]))


# 新增行
def insert_row(conn: sqlite3.Connection, *, sheet: str, data: dict[str, Any]) -> int:
    with transaction(conn) as cur:
        cur.execute(
            "INSERT INTO rows(sheet_name, data_json, created_at, updated_at) VALUES(?, ?, datetime('now'), datetime('now'))",
            (sheet, json.dumps(data, ensure_ascii=False)),
        )
        return int(cur.lastrowid)


# 更新行
def update_row(conn: sqlite3.Connection, *, row_id: int, data: dict[str, Any]) -> None:
    with transaction(conn) as cur:
        cur.execute(
            "UPDATE rows SET data_json=?, updated_at=datetime('now'), version=version+1 WHERE id=?",
            (json.dumps(data, ensure_ascii=False), row_id),
        )
        if cur.rowcount == 0:
            raise NotFoundError(f"Row {row_id} not found")


# 删除行
def delete_row(conn: sqlite3.Connection, *, row_id: int) -> bool:
    """Delete a row, returns True if deleted, False if not found."""
    with transaction(conn) as cur:
        cur.execute("DELETE FROM rows WHERE id=?", (row_id,))
        return cur.rowcount > 0


# 新增活动
def insert_event(
    conn: sqlite3.Connection,
    *,
    start_ts: int,
    start_text: str,
    location: str,
    nickname: str,
) -> int:
    with transaction(conn) as cur:
        cur.execute(
            """INSERT INTO events(start_ts, start_text, location, nickname, host_nickname, participants_json, version) 
               VALUES(?, ?, ?, ?, ?, ?, 1)""",
            (
                start_ts,
                start_text,
                location,
                nickname,
                nickname,
                json.dumps([nickname], ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)


# 删除活动
def delete_event(conn: sqlite3.Connection, *, event_id: int) -> bool:
    """Delete an event, returns True if deleted, False if not found."""
    with transaction(conn) as cur:
        cur.execute("DELETE FROM events WHERE id=?", (event_id,))
        return cur.rowcount > 0


# 列出未过期活动
def list_events_upcoming(conn: sqlite3.Connection, *, now_ts: int, until_ts: int) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, start_ts, start_text, location, nickname, host_nickname, participants_json, version
        FROM events
        WHERE start_ts >= ? AND start_ts <= ?
        ORDER BY start_ts ASC, id ASC
        """,
        (now_ts, until_ts),
    )
    out: list[dict[str, Any]] = []
    for r in cur.fetchall():
        d = dict(r)
        d["participants"] = json.loads(d.get("participants_json") or "[]")
        out.append(d)
    return out


# 按时间区间列出活动
def list_events_between(conn: sqlite3.Connection, *, start_ts: int, end_ts: int) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, start_ts, start_text, location, nickname, host_nickname, participants_json, version
        FROM events
        WHERE start_ts >= ? AND start_ts <= ?
        ORDER BY start_ts ASC, id ASC
        """,
        (start_ts, end_ts),
    )
    out: list[dict[str, Any]] = []
    for r in cur.fetchall():
        d = dict(r)
        d["participants"] = json.loads(d.get("participants_json") or "[]")
        out.append(d)
    return out


# 列出过期活动
def list_events_expired(conn: sqlite3.Connection, *, now_ts: int, limit: int = 5) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, start_ts, start_text, location, nickname, host_nickname, participants_json, version
        FROM events
        WHERE start_ts < ?
        ORDER BY start_ts DESC, id DESC
        LIMIT ?
        """,
        (now_ts, limit),
    )
    out: list[dict[str, Any]] = []
    for r in cur.fetchall():
        d = dict(r)
        d["participants"] = json.loads(d.get("participants_json") or "[]")
        out.append(d)
    return out


# 清理过期活动
def cleanup_expired_events(conn: sqlite3.Connection, *, now_ts: int, keep_limit: int = 5) -> int:
    """Delete expired events beyond the most recent keep_limit rows."""
    with transaction(conn) as cur:
        cur.execute(
            """
            DELETE FROM events
            WHERE id IN (
              SELECT id FROM events
              WHERE start_ts < ?
              ORDER BY start_ts DESC, id DESC
              LIMIT -1 OFFSET ?
            )
            """,
            (now_ts, keep_limit),
        )
        return cur.rowcount


# 获取活动
def get_event(conn: sqlite3.Connection, *, event_id: int) -> dict[str, Any] | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, start_ts, start_text, location, nickname, host_nickname, participants_json, version FROM events WHERE id=?",
        (event_id,),
    )
    r = cur.fetchone()
    if not r:
        return None
    d = dict(r)
    d["participants"] = json.loads(d.get("participants_json") or "[]")
    return d


# 安全更新参与者
def update_event_participants_safe(
    conn: sqlite3.Connection, 
    *, 
    event_id: int, 
    participants: list[str],
    expected_version: int
) -> bool:
    """
    Update event participants with optimistic locking.
    Returns True if update succeeded, False if version mismatch (concurrent modification).
    Raises NotFoundError if event doesn't exist.
    """
    with transaction(conn) as cur:
        cur.execute(
            """UPDATE events 
               SET participants_json=?, version=version+1 
               WHERE id=? AND version=?""",
            (json.dumps(participants, ensure_ascii=False), event_id, expected_version),
        )
        if cur.rowcount == 0:
            # Check if event exists
            cur.execute("SELECT 1 FROM events WHERE id=?", (event_id,))
            if cur.fetchone() is None:
                raise NotFoundError(f"Event {event_id} not found")
            return False  # Version mismatch
        return True


# 原子报名
def join_event_atomic(conn: sqlite3.Connection, *, event_id: int, nickname: str, now_ts: int) -> tuple[bool, str]:
    """
    Atomically join an event. Handles race conditions properly.
    Returns (success, message).
    """
    max_retries = 3
    for _ in range(max_retries):
        event = get_event(conn, event_id=event_id)
        if not event:
            return False, "活动不存在或已被删除"
        
        if int(event["start_ts"]) < now_ts:
            return False, "活动已过期，不能再报名"
        
        participants: list[str] = list(event.get("participants") or [])
        if nickname in participants:
            return True, "你已经报名了"
        
        participants.append(nickname)
        
        if update_event_participants_safe(
            conn, 
            event_id=event_id, 
            participants=participants,
            expected_version=event["version"]
        ):
            return True, "报名成功"
    
    return False, "服务器繁忙，请稍后重试"


# 原子取消报名
def leave_event_atomic(conn: sqlite3.Connection, *, event_id: int, nickname: str) -> tuple[bool, str]:
    """
    Atomically leave an event. Handles race conditions properly.
    Returns (success, message).
    """
    max_retries = 3
    for _ in range(max_retries):
        event = get_event(conn, event_id=event_id)
        if not event:
            return False, "活动不存在或已被删除"
        
        participants: list[str] = list(event.get("participants") or [])
        if nickname not in participants:
            return True, "你未报名该活动"
        
        participants = [p for p in participants if p != nickname]
        
        if update_event_participants_safe(
            conn, 
            event_id=event_id, 
            participants=participants,
            expected_version=event["version"]
        ):
            return True, "已取消报名"
    
    return False, "服务器繁忙，请稍后重试"


# Legacy function for backwards compatibility
# 兼容旧的参与者更新
def update_event_participants(conn: sqlite3.Connection, *, event_id: int, participants: list[str]) -> None:
    """Legacy function - prefer update_event_participants_safe for concurrent scenarios."""
    with transaction(conn) as cur:
        cur.execute(
            "UPDATE events SET participants_json=?, version=version+1 WHERE id=?",
            (json.dumps(participants, ensure_ascii=False), event_id),
        )


# 获取设置
def get_setting(conn: sqlite3.Connection, *, key: str) -> dict[str, Any] | None:
    cur = conn.cursor()
    cur.execute("SELECT key, value, version FROM settings WHERE key=?", (key,))
    r = cur.fetchone()
    if not r:
        return None
    return {"key": r["key"], "value": r["value"], "version": int(r["version"])}


# 安全更新设置
def update_setting_safe(
    conn: sqlite3.Connection,
    *,
    key: str,
    value: str,
    expected_version: int,
) -> bool:
    """
    Update a setting with optimistic locking.
    Returns True if update/insert succeeded, False on version mismatch.
    """
    with transaction(conn) as cur:
        if expected_version == 0:
            try:
                cur.execute(
                    "INSERT INTO settings(key, value, version, updated_at) VALUES(?, ?, 1, datetime('now'))",
                    (key, value),
                )
                return True
            except sqlite3.IntegrityError:
                return False

        cur.execute(
            """
            UPDATE settings
            SET value=?, version=version+1, updated_at=datetime('now')
            WHERE key=? AND version=?
            """,
            (value, key, expected_version),
        )
        if cur.rowcount > 0:
            return True

        # If setting doesn't exist, allow insert (e.g. after DB reset).
        cur.execute("SELECT 1 FROM settings WHERE key=?", (key,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO settings(key, value, version, updated_at) VALUES(?, ?, 1, datetime('now'))",
                (key, value),
            )
            return True
        return False

# ============ Row Images ============

def insert_row_image(conn: sqlite3.Connection, *, row_id: int, filename: str, original_name: str) -> int:
    with transaction(conn) as cur:
        cur.execute(
            "INSERT INTO row_images(row_id, filename, original_name) VALUES(?, ?, ?)",
            (row_id, filename, original_name),
        )
        return int(cur.lastrowid)


def list_row_images(conn: sqlite3.Connection, *, row_id: int) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, row_id, filename, original_name, created_at FROM row_images WHERE row_id=? ORDER BY id",
        (row_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def list_rows_images(conn: sqlite3.Connection, *, row_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    """Batch fetch images for multiple rows."""
    if not row_ids:
        return {}
    placeholders = ",".join("?" for _ in row_ids)
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, row_id, filename, original_name, created_at FROM row_images WHERE row_id IN ({placeholders}) ORDER BY row_id, id",
        row_ids,
    )
    result: dict[int, list[dict[str, Any]]] = {}
    for r in cur.fetchall():
        d = dict(r)
        rid = int(d["row_id"])
        result.setdefault(rid, []).append(d)
    return result


def delete_row_image(conn: sqlite3.Connection, *, image_id: int) -> str | None:
    """Delete a row image record. Returns filename if deleted, None if not found."""
    cur = conn.cursor()
    cur.execute("SELECT filename FROM row_images WHERE id=?", (image_id,))
    r = cur.fetchone()
    if not r:
        return None
    filename = r["filename"]
    with transaction(conn) as tcur:
        tcur.execute("DELETE FROM row_images WHERE id=?", (image_id,))
    return filename


def get_row_image(conn: sqlite3.Connection, *, image_id: int) -> dict[str, Any] | None:
    cur = conn.cursor()
    cur.execute("SELECT id, row_id, filename, original_name, created_at FROM row_images WHERE id=?", (image_id,))
    r = cur.fetchone()
    if not r:
        return None
    return dict(r)

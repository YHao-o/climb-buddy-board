import db as guide_db


def test_insert_and_list_activities_desc(conn):
    guide_db.insert_activity(conn, event_id=1, action="create", actor="A",
                             location="香蕉", start_text="2026-06-27 10:30", ts=100)
    guide_db.insert_activity(conn, event_id=1, action="join", actor="B",
                             location="香蕉", start_text="2026-06-27 10:30", ts=200)
    acts = guide_db.list_activities(conn, limit=10)
    assert [a["action"] for a in acts] == ["join", "create"]  # ts 倒序
    assert acts[0]["actor"] == "B"


def test_prune_activities(conn):
    guide_db.insert_activity(conn, event_id=None, action="create", actor="A",
                             location="x", start_text="y", ts=100)
    guide_db.insert_activity(conn, event_id=None, action="create", actor="A",
                             location="x", start_text="y", ts=500)
    removed = guide_db.prune_activities(conn, cutoff_ts=300)
    assert removed == 1
    assert len(guide_db.list_activities(conn, limit=10)) == 1


def test_cleanup_expired_events_by_cutoff(conn):
    guide_db.insert_event(conn, start_ts=100, start_text="old", location="A", nickname="n")
    guide_db.insert_event(conn, start_ts=999, start_text="new", location="B", nickname="n")
    removed = guide_db.cleanup_expired_events(conn, cutoff_ts=500)
    assert removed == 1
    remaining = guide_db.list_events_between(conn, start_ts=0, end_ts=10_000)
    assert [e["start_text"] for e in remaining] == ["new"]


def test_backfill_idempotent(conn):
    guide_db.insert_event(conn, start_ts=100, start_text="2026-06-01 08:00",
                          location="A", nickname="host")
    n1 = guide_db.backfill_activities_if_needed(conn)
    n2 = guide_db.backfill_activities_if_needed(conn)
    assert n1 == 1
    assert n2 == 0  # 第二次不再补
    acts = guide_db.list_activities(conn, limit=10)
    assert acts[0]["action"] == "create" and acts[0]["actor"] == "host"

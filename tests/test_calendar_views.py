from datetime import date

import calendar_views as cv


def test_classify_period_boundaries():
    assert cv.classify_period(5) == "morning"
    assert cv.classify_period(11) == "morning"
    assert cv.classify_period(12) == "afternoon"
    assert cv.classify_period(17) == "afternoon"
    assert cv.classify_period(18) == "evening"
    assert cv.classify_period(4) == "evening"
    assert cv.classify_period(23) == "evening"


def test_start_of_week_is_monday():
    # 2026-06-26 是周五 -> 周一应为 2026-06-22
    assert cv.start_of_week(date(2026, 6, 26)) == date(2026, 6, 22)
    assert cv.start_of_week(date(2026, 6, 22)) == date(2026, 6, 22)


def test_build_week_groups_by_day():
    from datetime import datetime
    # 周一 2026-06-22 09:00
    mon = int(datetime(2026, 6, 22, 9, 0).timestamp())
    events = [{"id": 1, "start_ts": mon, "start_text": "2026-06-22 09:00", "location": "香蕉"}]
    today = date(2026, 6, 22)
    week = cv.build_week(events, today=today, now_ts=mon)
    assert len(week) == 7
    assert week[0]["is_today"] is True
    assert len(week[0]["items"]) == 1
    it = week[0]["items"][0]
    assert it["time_text"] == "09:00"
    assert it["period"] == "morning"
    assert it["period_label"] == "上午"
    assert it["detail_url"] == "/events/1"
    assert it["is_active"] is True
    assert len(week[1]["items"]) == 0


def test_build_month_has_six_weeks_and_anchors():
    today = date(2026, 6, 26)
    m = cv.build_month([], year=2026, month=6, today=today)
    assert m["title"] == "2026年6月"
    assert m["prev_anchor"] == "2026-05"
    assert m["next_anchor"] == "2026-07"
    assert m["weekday_headers"] == ["一", "二", "三", "四", "五", "六", "日"]
    assert len(m["weeks"]) == 6
    for w in m["weeks"]:
        assert len(w) == 7


def test_build_month_marks_in_month_and_today():
    today = date(2026, 6, 26)
    m = cv.build_month([], year=2026, month=6, today=today)
    cells = [c for w in m["weeks"] for c in w]
    # 2026-06 第一天是周一，所以首格就是 6-01，in_month=True
    first = cells[0]
    assert first["date"] == "2026-06-01"
    assert first["in_month"] is True
    today_cells = [c for c in cells if c["date"] == "2026-06-26"]
    assert len(today_cells) == 1 and today_cells[0]["is_today"] is True
    # 末尾应含下月日期且 in_month=False
    assert any(c["in_month"] is False for c in cells)


def test_build_month_overflow_counts():
    from datetime import datetime
    events = []
    for h in (9, 13, 19, 20):  # 同一天 4 个活动
        ts = int(datetime(2026, 6, 10, h, 0).timestamp())
        events.append({"id": h, "start_ts": ts, "start_text": f"2026-06-10 {h}:00", "location": "X"})
    m = cv.build_month(events, year=2026, month=6, today=date(2026, 6, 1))
    cell = [c for w in m["weeks"] for c in w if c["date"] == "2026-06-10"][0]
    assert cell["total"] == 4
    assert len(cell["visible_items"]) == cv.MONTH_MAX_LINES  # 2
    assert len(cell["dots"]) == cv.MONTH_MAX_DOTS  # 3
    assert cell["more_count"] == 4 - cv.MONTH_MAX_LINES  # 2


def test_build_list_groups_sorted():
    from datetime import datetime
    e1 = {"id": 1, "start_ts": int(datetime(2026, 6, 27, 13, 30).timestamp()),
          "start_text": "2026-06-27 13:30", "location": "B"}
    e2 = {"id": 2, "start_ts": int(datetime(2026, 6, 27, 10, 30).timestamp()),
          "start_text": "2026-06-27 10:30", "location": "A"}
    e3 = {"id": 3, "start_ts": int(datetime(2026, 6, 26, 19, 0).timestamp()),
          "start_text": "2026-06-26 19:00", "location": "C"}
    groups = cv.build_list([e1, e2, e3], today=date(2026, 6, 27))
    assert [g["date"] for g in groups] == ["2026-06-26", "2026-06-27"]
    assert groups[0]["is_past"] is True
    assert groups[1]["is_today"] is True
    # 组内按时间升序
    assert [e["id"] for e in groups[1]["events"]] == [2, 1]
    assert groups[1]["events"][0]["time_text"] == "10:30"
    assert groups[1]["title"] == "6.27（周六）"


def test_build_heatmap_empty():
    h = cv.build_heatmap([])
    assert h["max_count"] == 0
    assert h["summary"] == "暂无活动数据"
    assert len(h["rows"]) == 7
    assert all(len(r["cells"]) == 3 for r in h["rows"])
    assert all(c["level"] == 0 for r in h["rows"] for c in r["cells"])


def test_build_heatmap_counts_and_summary():
    import math
    from datetime import datetime
    events = []
    # 周六(2026-06-27)上午 3 个
    for hh in (9, 10, 11):
        events.append({"id": hh, "start_ts": int(datetime(2026, 6, 27, hh, 0).timestamp()),
                       "start_text": "x", "location": "x"})
    # 周一(2026-06-22)晚上 1 个
    events.append({"id": 99, "start_ts": int(datetime(2026, 6, 22, 20, 0).timestamp()),
                   "start_text": "x", "location": "x"})
    h = cv.build_heatmap(events)
    assert h["max_count"] == 3
    sat = [r for r in h["rows"] if r["weekday_index"] == 5][0]
    sat_morning = [c for c in sat["cells"] if c["period"] == "morning"][0]
    assert sat_morning["count"] == 3
    assert sat_morning["level"] == 4
    mon = [r for r in h["rows"] if r["weekday_index"] == 0][0]
    mon_evening = [c for c in mon["cells"] if c["period"] == "evening"][0]
    assert mon_evening["count"] == 1
    assert mon_evening["level"] == math.ceil(1 / 3 * 4)  # 2
    assert h["summary"] == "周六上午最热门"


def test_build_feed_basic():
    now = 1_000_000
    acts = [
        {"action": "create", "actor": "海鑫", "location": "香蕉(华侨城)",
         "start_text": "2026-06-27 10:30", "ts": now - 120},
        {"action": "join", "actor": "seisei", "location": "upper南光",
         "start_text": "2026-06-26 19:00", "ts": now - 3600},
        {"action": "leave", "actor": "X", "location": "CUBE罗湖",
         "start_text": "bad-format", "ts": now - 86400 * 2},
    ]
    feed = cv.build_feed(acts, now_ts=now)
    assert feed[0]["icon"] == "🟢" and feed[0]["verb"] == "发起了"
    assert feed[0]["when"] == "6.27"
    assert feed[0]["relative_time"] == "2分钟前"
    assert feed[1]["verb"] == "报名了" and feed[1]["relative_time"] == "1小时前"
    assert feed[2]["icon"] == "🔴" and feed[2]["when"] == "bad-format"
    assert feed[2]["relative_time"] == "2天前"


def test_build_feed_just_now():
    now = 500
    acts = [{"action": "delete", "actor": "Y", "location": "Z",
             "start_text": "2026-06-01 08:00", "ts": now - 10}]
    feed = cv.build_feed(acts, now_ts=now)
    assert feed[0]["relative_time"] == "刚刚"
    assert feed[0]["icon"] == "🗑️"





import db as guide_db


def test_db_imports_and_inits(conn):
    assert guide_db.has_any_data(conn) is False

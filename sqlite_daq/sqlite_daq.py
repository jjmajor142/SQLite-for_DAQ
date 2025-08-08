"""
Refactored SQLite wrapper optimized for DAQ-style, append-only logging.
Thread-safe per-connection usage, with helpers for CSV export and column selection.
"""

from __future__ import annotations

import csv
import sqlite3
import threading
import time
from typing import Dict, Iterable, List, Sequence


def _quote_ident(name: str) -> str:
    """
    Quote an identifier (table/column) for SQLite to avoid issues with
    reserved keywords and special characters.
    """
    # Double any internal double quotes and wrap the name in quotes
    return f"\"{name.replace('"', '""')}\""


class SQLiteDaqWrapper:
    """
    Tiny convenience layer around sqlite3 to simplify common DAQ logging tasks.
    Each instance owns its own connection. Use one instance per thread.
    """

    def __init__(self, db_name: str) -> None:
        # A connection must be used only from its creating thread by default.
        # We keep that invariant by creating a separate wrapper in worker threads.
        if not db_name.lower().endswith('.db'):
            db_name += '.db'
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.lock = threading.Lock()

        # Safer/concurrent-friendly defaults
        with self.lock:
            self.cursor.execute("PRAGMA journal_mode=WAL;")
            self.cursor.execute("PRAGMA synchronous=NORMAL;")
            self.conn.commit()

    # ---------- schema ----------

    def create_table_from_dict(self, table_name: str, input_dict: Dict[str, object]) -> None:
        """
        Create a table with columns inferred from the types of `input_dict` values.
        - int/float -> INTEGER/REAL
        - otherwise -> TEXT
        """
        cols: List[str] = ['"id" INTEGER PRIMARY KEY']
        for key, value in input_dict.items():
            if value is int or isinstance(value, int):
                sql_type = "INTEGER"
            elif value is float or isinstance(value, float):
                sql_type = "REAL"
            else:
                sql_type = "TEXT"
            cols.append(f"{_quote_ident(key)} {sql_type}")

        query = f"CREATE TABLE {_quote_ident(table_name)} ({', '.join(cols)})"
        try:
            with self.lock:
                self.cursor.execute(query)
                self.conn.commit()
        except sqlite3.OperationalError as exc:
            # e.g., table already exists
            print(exc)

    # ---------- writes ----------

    def append_data_to_table(self, table_name: str, input_dict: Dict[str, object]) -> None:
        """
        Insert a single row using the keys/values from `input_dict`.
        If the table does not exist, it will be created using the dict shape.
        """
        columns = ", ".join(_quote_ident(k) for k in input_dict.keys())
        placeholders = ", ".join(["?"] * len(input_dict))
        query = f"INSERT INTO {_quote_ident(table_name)} ({columns}) VALUES ({placeholders})"
        try:
            with self.lock:
                self.cursor.execute(query, tuple(input_dict.values()))
                self.conn.commit()
        except sqlite3.OperationalError as exc:
            # Table doesn't exist: create and retry
            if "no such table" in str(exc):
                self.create_table_from_dict(table_name, input_dict)
                self.append_data_to_table(table_name, input_dict)
            else:
                print(exc)

    # ---------- reads ----------

    def fetch_all(self, table_name: str) -> List[Dict[str, object]]:
        """
        Fetch all rows from a table as a list of dicts.
        """
        with self.lock:
            self.cursor.execute(f"SELECT * FROM {_quote_ident(table_name)}")
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def fetch_columns(self, table_name: str, columns: Sequence[str]) -> List[Dict[str, object]]:
        """
        Fetch specific columns from a table as a list of dicts.
        """
        col_str = ", ".join(_quote_ident(col) for col in columns)
        with self.lock:
            self.cursor.execute(f"SELECT {col_str} FROM {_quote_ident(table_name)}")
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    # ---------- export ----------

    def export_to_csv(self, table_name: str, csv_path: str) -> None:
        """
        Export all rows from a table to a CSV file.
        """
        rows = self.fetch_all(table_name)
        if not rows:
            return
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def close(self) -> None:
        """
        Close the database connection.
        """
        with self.lock:
            self.conn.close()

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
    return f"\"{name.replace('\"', '\"\"')}\""


class SQLiteDaqWrapper:
    """
    Tiny convenience layer around sqlite3 to simplify common DAQ logging tasks.
    Each instance owns its own connection. Use one instance per thread.
    """

    def __init__(self, db_name: str) -> None:
        # A connection must be used only from its creating thread by default.
        # We keep that invariant by creating a separate wrapper in worker threads.
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
        query = (
            f"INSERT INTO {_quote_ident(table_name)} ({columns}) "
            f"VALUES ({placeholders})"
        )
        values = tuple(input_dict.values())

        try:
            with self.lock:
                self.cursor.execute(query, values)
                self.conn.commit()
        except sqlite3.OperationalError:
            # Likely missing table; create and retry once
            self.create_table_from_dict(table_name, input_dict)
            with self.lock:
                self.cursor.execute(query, values)
                self.conn.commit()

    # ---------- reads ----------

    def get_last_n_rows(self, table_name: str, n: int) -> Dict[str, list]:
        """
        Return a dict of column->list for the last `n` rows (descending id).
        """
        query = f"SELECT * FROM {_quote_ident(table_name)} ORDER BY id DESC LIMIT {int(n)}"
        with self.lock:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            col_names = [d[0] for d in self.cursor.description]

        data: Dict[str, list] = {name: [] for name in col_names}
        for row in rows:
            for name, val in zip(col_names, row):
                data[name].append(val)
        return data

    def get_n_rows_from_column(self, table_name: str, column_name: str, n: int) -> Dict[str, list]:
        """
        Return a dict with a single column->list for the last `n` rows.
        """
        query = (
            f"SELECT {_quote_ident(column_name)} FROM {_quote_ident(table_name)} "
            f"ORDER BY id DESC LIMIT {int(n)}"
        )
        with self.lock:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            col_names = [d[0] for d in self.cursor.description]

        data: Dict[str, list] = {name: [r[0] for r in rows] for name in col_names}
        return data

    def get_n_rows_from_columns(self, table_name: str, column_names: Sequence[str], n: int) -> Dict[str, list]:
        """
        Return dict of selected columns for the last `n` rows, in chronological order.
        """
        cols = ", ".join(_quote_ident(c) for c in column_names)
        query = (
            f"SELECT {cols} FROM {_quote_ident(table_name)} "
            f"ORDER BY id DESC LIMIT {int(n)}"
        )
        with self.lock:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            col_names = [d[0] for d in self.cursor.description]

        data: Dict[str, list] = {}
        for name, col in zip(col_names, zip(*rows) if rows else [[]] * len(col_names)):
            seq = list(col)
            seq.reverse()  # chronological
            data[name] = seq
        return data

    def get_all_rows_from_columns(self, table_name: str, column_names: Sequence[str]) -> Dict[str, list]:
        """
        Return dict of selected columns for the full table in chronological order.
        """
        cols = ", ".join(_quote_ident(c) for c in column_names)
        query = f"SELECT {cols} FROM {_quote_ident(table_name)}"
        with self.lock:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            col_names = [d[0] for d in self.cursor.description]

        data: Dict[str, list] = {}
        for name, col in zip(col_names, zip(*rows) if rows else [[]] * len(col_names)):
            data[name] = list(col)
        return data

    # ---------- csv ----------

    def sqlite_to_csv(self, table_name: str, csv_filename: str, n: int | None = None) -> None:
        """
        Export the table (optionally last n rows) to a CSV file.
        """
        if n is None:
            query = f"SELECT * FROM {_quote_ident(table_name)}"
        else:
            query = f"SELECT * FROM {_quote_ident(table_name)} ORDER BY id DESC LIMIT {int(n)}"

        with self.lock:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            headers = [d[0] for d in self.cursor.description]

        # If `n` was provided with DESC ordering, write back in chronological order.
        if n is not None:
            rows.reverse()

        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def create_table_from_csv(self, table_name: str, csv_filename: str) -> None:
        """
        Create a table with TEXT columns from a CSV and import the rows.
        """
        with open(csv_filename, "r", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            cols = ", ".join(f'{_quote_ident(h)} TEXT' for h in headers)
            create_query = f"CREATE TABLE {_quote_ident(table_name)} ({cols})"

            with self.lock:
                self.cursor.execute(create_query)

            insert_query = (
                f"INSERT INTO {_quote_ident(table_name)} "
                f"VALUES ({', '.join(['?'] * len(headers))})"
            )
            for row in reader:
                with self.lock:
                    self.cursor.execute(insert_query, row)
            self.conn.commit()

    # ---------- utils ----------

    def reset_table_data(self, table_name: str) -> None:
        """Delete all rows from the given table."""
        with self.lock:
            self.cursor.execute(f"DELETE FROM {_quote_ident(table_name)}")
            self.conn.commit()

    def close(self) -> None:
        """Close the cursor and connection."""
        with self.lock:
            self.cursor.close()
            self.conn.close()

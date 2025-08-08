# SQLite-DAQ

A SQLite wrapper optimized for DAQ-style, append-only logging. Thread-safe per-connection usage, with helpers for CSV export and column selection.

## Installation

```bash
pip install sqlite-daq
```

## Usage

```python
from sqlite_daq import SQLiteDaqWrapper

db = SQLiteDaqWrapper('mydata')
db.append_data_to_table('log', {'value': 42, 'timestamp': '2025-08-08'})
rows = db.fetch_all('log')
print(rows)
db.export_to_csv('log', 'log.csv')
db.close()
```

## License
MIT

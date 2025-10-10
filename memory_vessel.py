"""AI Memory Containment Vessel
================================

This module implements a scalable memory containment vessel designed for AI
systems. The vessel balances fast access to recent memories with the ability to
scale storage seamlessly as usage grows. It combines an in-memory cache for
quick lookups with an automatically sharded SQLite backing store.

Usage example (CLI)::

    python memory_vessel.py store --id greeting --data "Hello, world!"
    python memory_vessel.py retrieve --id greeting
    python memory_vessel.py stats

Run ``python memory_vessel.py --help`` for a full list of commands.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class MemoryRecord:
    """Representation of a single memory entry."""

    memory_id: str
    data: str
    metadata: Dict[str, object]
    created_at: float


class LRUCache:
    """A minimal LRU cache implementation using :class:`OrderedDict`."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._store: "OrderedDict[str, MemoryRecord]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[MemoryRecord]:
        with self._lock:
            record = self._store.get(key)
            if record is not None:
                # Mark as recently used
                self._store.move_to_end(key)
            return record

    def put(self, key: str, value: MemoryRecord) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class MemoryContainmentVessel:
    """Scalable memory storage that shards automatically as usage grows."""

    def __init__(
        self,
        db_path: str = "memory_vessel.db",
        cache_size: int = 256,
        shard_capacity: int = 10_000,
    ) -> None:
        self.db_path = db_path
        self.cache = LRUCache(cache_size)
        self.shard_capacity = max(1, shard_capacity)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------
    def _initialize_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shard_registry (
                    shard_name TEXT PRIMARY KEY,
                    record_count INTEGER NOT NULL
                )
                """
            )
        self._ensure_shard(0)

    def _ensure_shard(self, index: int) -> str:
        shard_name = f"mem_shard_{index:05d}"
        with self._conn:
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {shard_name} (
                    memory_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                INSERT OR IGNORE INTO shard_registry (shard_name, record_count)
                VALUES (?, 0)
                """,
                (shard_name,),
            )
        return shard_name

    def _load_shards(self) -> List[Tuple[str, int]]:
        cursor = self._conn.execute(
            "SELECT shard_name, record_count FROM shard_registry ORDER BY shard_name"
        )
        return [(row["shard_name"], row["record_count"]) for row in cursor]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def store(
        self,
        memory_id: str,
        data: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> MemoryRecord:
        metadata = metadata or {}
        created_at = time.time()
        encoded_metadata = json.dumps(metadata, sort_keys=True)
        record = MemoryRecord(memory_id, data, metadata, created_at)

        with self._lock, self._conn:
            shard_name = self._locate_shard_for_memory(memory_id)
            if shard_name:
                self._conn.execute(
                    f"""
                    UPDATE {shard_name}
                    SET data = ?, metadata = ?, created_at = ?
                    WHERE memory_id = ?
                    """,
                    (data, encoded_metadata, created_at, memory_id),
                )
            else:
                shard_name = self._select_shard_for_write()
                self._conn.execute(
                    f"""
                    INSERT INTO {shard_name} (memory_id, data, metadata, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (memory_id, data, encoded_metadata, created_at),
                )
                self._increment_shard_count(shard_name)
            self._purge_duplicate_records(memory_id, shard_name)
        self.cache.put(memory_id, record)
        return record

    def retrieve(self, memory_id: str) -> Optional[MemoryRecord]:
        cached = self.cache.get(memory_id)
        if cached:
            return cached

        with self._lock:
            for shard_name, _ in self._load_shards():
                cursor = self._conn.execute(
                    f"SELECT memory_id, data, metadata, created_at FROM {shard_name} WHERE memory_id = ?",
                    (memory_id,),
                )
                row = cursor.fetchone()
                if row:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                    record = MemoryRecord(
                        row["memory_id"], row["data"], metadata, row["created_at"]
                    )
                    self.cache.put(memory_id, record)
                    return record
        return None

    def delete(self, memory_id: str) -> bool:
        removed = False
        with self._lock, self._conn:
            for shard_name, _ in self._load_shards():
                result = self._conn.execute(
                    f"DELETE FROM {shard_name} WHERE memory_id = ?",
                    (memory_id,),
                )
                if result.rowcount:
                    self._decrement_shard_count(shard_name)
                    removed = True
        if removed:
            self.cache.invalidate(memory_id)
        return removed

    def search(self, text: str, limit: int = 10) -> Iterator[MemoryRecord]:
        pattern = f"%{text}%"
        with self._lock:
            for shard_name, _ in self._load_shards():
                cursor = self._conn.execute(
                    f"""
                    SELECT memory_id, data, metadata, created_at
                    FROM {shard_name}
                    WHERE data LIKE ? OR metadata LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (pattern, pattern, limit),
                )
                for row in cursor:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                    yield MemoryRecord(
                        row["memory_id"], row["data"], metadata, row["created_at"]
                    )

    def purge(self) -> None:
        with self._lock, self._conn:
            for shard_name, _ in self._load_shards():
                self._conn.execute(f"DROP TABLE IF EXISTS {shard_name}")
            self._conn.execute("DELETE FROM shard_registry")
        self.cache.clear()
        self._ensure_shard(0)

    def stats(self) -> Dict[str, object]:
        shards = self._load_shards()
        total_records = sum(count for _, count in shards)
        return {
            "db_path": os.path.abspath(self.db_path),
            "shard_capacity": self.shard_capacity,
            "total_records": total_records,
            "shards": [
                {"name": name, "records": count, "utilization": count / self.shard_capacity}
                for name, count in shards
            ],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_shard_for_write(self) -> str:
        shards = self._load_shards()
        if not shards:
            return self._ensure_shard(0)
        shard_name, count = shards[-1]
        if count >= self.shard_capacity:
            shard_index = len(shards)
            shard_name = self._ensure_shard(shard_index)
        return shard_name

    def _increment_shard_count(self, shard_name: str) -> None:
        self._conn.execute(
            """
            UPDATE shard_registry
            SET record_count = record_count + 1
            WHERE shard_name = ?
            """,
            (shard_name,),
        )

    def _decrement_shard_count(self, shard_name: str) -> None:
        self._conn.execute(
            """
            UPDATE shard_registry
            SET record_count = MAX(record_count - 1, 0)
            WHERE shard_name = ?
            """,
            (shard_name,),
        )

    def _locate_shard_for_memory(self, memory_id: str) -> Optional[str]:
        for shard_name, _ in self._load_shards():
            cursor = self._conn.execute(
                f"SELECT 1 FROM {shard_name} WHERE memory_id = ?",
                (memory_id,),
            )
            if cursor.fetchone():
                return shard_name
        return None

    def _purge_duplicate_records(self, memory_id: str, authoritative_shard: str) -> None:
        for shard_name, _ in self._load_shards():
            if shard_name == authoritative_shard:
                continue
            result = self._conn.execute(
                f"DELETE FROM {shard_name} WHERE memory_id = ?",
                (memory_id,),
            )
            if result.rowcount:
                self._decrement_shard_count(shard_name)


def _cmd_store(args: argparse.Namespace) -> None:
    vessel = MemoryContainmentVessel(args.database, cache_size=args.cache_size, shard_capacity=args.shard_capacity)
    try:
        metadata = json.loads(args.metadata) if args.metadata else None
        record = vessel.store(args.id, args.data, metadata)
        print(f"Stored memory {record.memory_id} at {time.ctime(record.created_at)}")
    finally:
        vessel.close()


def _cmd_retrieve(args: argparse.Namespace) -> None:
    vessel = MemoryContainmentVessel(args.database, cache_size=args.cache_size, shard_capacity=args.shard_capacity)
    try:
        record = vessel.retrieve(args.id)
        if record:
            print(json.dumps(record.__dict__, indent=2))
        else:
            print(f"Memory {args.id!r} not found.")
    finally:
        vessel.close()


def _cmd_search(args: argparse.Namespace) -> None:
    vessel = MemoryContainmentVessel(args.database, cache_size=args.cache_size, shard_capacity=args.shard_capacity)
    try:
        for record in vessel.search(args.query, args.limit):
            print(json.dumps(record.__dict__, indent=2))
    finally:
        vessel.close()


def _cmd_delete(args: argparse.Namespace) -> None:
    vessel = MemoryContainmentVessel(args.database, cache_size=args.cache_size, shard_capacity=args.shard_capacity)
    try:
        if vessel.delete(args.id):
            print(f"Deleted memory {args.id!r}.")
        else:
            print(f"Memory {args.id!r} not found.")
    finally:
        vessel.close()


def _cmd_stats(args: argparse.Namespace) -> None:
    vessel = MemoryContainmentVessel(args.database, cache_size=args.cache_size, shard_capacity=args.shard_capacity)
    try:
        print(json.dumps(vessel.stats(), indent=2))
    finally:
        vessel.close()


def _cmd_purge(args: argparse.Namespace) -> None:
    vessel = MemoryContainmentVessel(args.database, cache_size=args.cache_size, shard_capacity=args.shard_capacity)
    try:
        vessel.purge()
        print("All shards purged and schema reset.")
    finally:
        vessel.close()


def _cmd_demo(args: argparse.Namespace) -> None:
    vessel = MemoryContainmentVessel(args.database, cache_size=args.cache_size, shard_capacity=args.shard_capacity)
    try:
        vessel.purge()
        print("Running demo... creating 12 memories with shard capacity of 5")
        vessel.shard_capacity = 5
        for i in range(12):
            vessel.store(f"memory-{i}", f"payload-{i}", {"index": i})
        print(json.dumps(vessel.stats(), indent=2))
        print("Retrieving memory-3:")
        record = vessel.retrieve("memory-3")
        if record:
            print(json.dumps(record.__dict__, indent=2))
    finally:
        vessel.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Memory Containment Vessel")
    parser.add_argument("--database", default="memory_vessel.db", help="Path to the SQLite database file.")
    parser.add_argument("--cache-size", type=int, default=256, help="LRU cache capacity for hot memories.")
    parser.add_argument(
        "--shard-capacity",
        type=int,
        default=10_000,
        help="Maximum records per shard before a new shard is created.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    store_parser = subparsers.add_parser("store", help="Store or update a memory entry.")
    store_parser.add_argument("--id", required=True, help="Memory identifier.")
    store_parser.add_argument("--data", required=True, help="Memory payload.")
    store_parser.add_argument("--metadata", help="JSON metadata for the memory.")
    store_parser.set_defaults(func=_cmd_store)

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve a memory entry.")
    retrieve_parser.add_argument("--id", required=True, help="Memory identifier.")
    retrieve_parser.set_defaults(func=_cmd_retrieve)

    search_parser = subparsers.add_parser("search", help="Search across memory payloads and metadata.")
    search_parser.add_argument("query", help="Substring to search for.")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum records to return per shard.")
    search_parser.set_defaults(func=_cmd_search)

    delete_parser = subparsers.add_parser("delete", help="Delete a memory entry.")
    delete_parser.add_argument("--id", required=True, help="Memory identifier.")
    delete_parser.set_defaults(func=_cmd_delete)

    stats_parser = subparsers.add_parser("stats", help="Show vessel statistics.")
    stats_parser.set_defaults(func=_cmd_stats)

    purge_parser = subparsers.add_parser("purge", help="Erase all memories and reset the vessel.")
    purge_parser.set_defaults(func=_cmd_purge)

    demo_parser = subparsers.add_parser("demo", help="Run a demonstration of automatic sharding.")
    demo_parser.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

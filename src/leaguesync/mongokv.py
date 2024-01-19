

from pymongo import MongoClient
from typing import Any

class MongoKV:
    """Key Value store backed by MongoDB"""
    def __init__(self, collection):

        self.collection = collection

        self.collection.create_index("key", unique=True)

    def set(self, k: str, v: Any) -> None:
        # Write or update the key/value document in the collection
        self.collection.update_one(
            {"key": k },  # condition
            {"$set": {"value": v}},  # new value
            upsert=True  # insert if not exists, update if exists
        )

    def get(self, k: str) -> Any:

        doc = self.collection.find_one({"key": {"$eq": k}})
        return doc['value'] if doc else None

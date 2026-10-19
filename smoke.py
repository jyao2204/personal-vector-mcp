import os
import httpx

EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:8080")
URL = f"{EMBED_URL}/v1/embeddings"


def post(payload):
    return httpx.post(URL, json=payload, timeout=300)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


health = httpx.get(f"{EMBED_URL}/health").json()
assert health["dim"] == 1024, health

docs = [
    "Invoices go out by email on the first business day of each month.",
    "The sourdough starter needs feeding every twelve hours.",
]
d = [e["embedding"] for e in post({"input": docs}).json()["data"]]
q = post({"input": "when are invoices sent", "input_type": "query"}).json()["data"][0]["embedding"]
assert all(len(v) == 1024 for v in d + [q])
assert abs(dot(q, q) - 1.0) < 1e-3, "not normalised"
assert dot(q, d[0]) > dot(q, d[1]), "query did not prefer the relevant document"

plain = post({"input": "when are invoices sent"}).json()["data"][0]["embedding"]
assert dot(q, plain) < 0.9999, "input_type=query had no effect"

assert post({"input": []}).status_code == 422
assert post({"input": ""}).status_code == 422
assert post({"input": [""]}).status_code == 422
assert post({"input": ["a", ""]}).status_code == 422

big = post({"input": "word " * 20000}).json()["data"][0]["embedding"]
assert len(big) == 1024

print("smoke ok")

from embed_client import embed

many = embed([f"note number {i}" for i in range(70)])
assert len(many) == 70
assert dot(many[0], embed(["note number 0"])[0]) > 0.999, "batching broke order or values"
assert dot(many[69], embed(["note number 69"])[0]) > 0.999
assert len(embed(["x"], input_type="query")[0]) == 1024
print("client ok")

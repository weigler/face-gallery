import os
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from insightface.app import FaceAnalysis

API_KEY = os.getenv("API_KEY")
ALBUM = os.getenv("ALBUM")

ALBUMS = {
  "RPM2406": "1-LyABC7nFLJ9M1j3k1iZHT0LxAYoqje2",
  "RPM2407": "1-It0lAedNjHY4lS0nUU1EXxIgpLn3L_Z",
  "RPM2410": "1-D_FwO4HKzbWITv3ZRnVpozLe2LUGEh8",
  "RPM2411": "1-yPDxcJkwYxJcTCWEC3chbCQF-eFMv0V",
  "RPM2412": "1-xWf_2z6xqXqREnqsTQ0JQfndD-8TW5D",
  "RPM2501": "1-TL7LLUmmbEfrTvZDUp7zC7gDuf7TOrv",
  "RPM2502": "1FL2x8eeN_2tTHRFEi7f4oetHSWvKnhNd",
  "RPM2503": "1FLAHTI74-PhoreWcQcfDtcLtDYLiD-2J",
  "RPM2504": "1iu0SS8JCpAmpb-MXfrJH8Jj7qHdMwrOe",
  "RPM2505": "1HREu2ddVkgfRcWP2SwobLzuf4xq4uaAs",
  "RPM2506": "10btbePXG2PYEkz5rYxKX2sOyjCSNRjuL",
  "RPM2601": "1DtvBHNkBXS7FyzkF_lutZrerY6S7rDkF",
  "GISELA60": "1Wz5I-D-K7wLwP9ExMUdawXkkAK-i2KXT",
  "ALTORIO26": "1hMHJ9kKDb-PPwcOYOfqNwaAgKMd54lMT",
}

FOLDER_ID = ALBUMS.get(ALBUM)

if not FOLDER_ID:
    raise Exception("Álbum inválido")

# 🔥 modelo profissional
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

def normalize(v):
    return (v / np.linalg.norm(v)).tolist()

def get_files():
    query = f"'{FOLDER_ID}' in parents and mimeType contains 'image/' and trashed=false"
    url = f"https://www.googleapis.com/drive/v3/files?q={query}&fields=files(id,name)&pageSize=1000&key={API_KEY}"
    res = requests.get(url).json()
    return res.get("files", [])

def get_embedding(url):
    try:
        img = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
        faces = app.get(img)

        if not faces:
            return None

        emb = faces[0].embedding

        # 🔥 normalização
        return normalize(emb)

    except:
        return None

def cluster(data):
    groups = []

    for item in data:
        added = False

        for g in groups:
            avg = np.mean([x["descriptor"] for x in g], axis=0)
            dist = np.linalg.norm(np.array(item["descriptor"]) - avg)

            if dist < 0.6:
                g.append(item)
                added = True
                break

        if not added:
            groups.append([item])

    return groups

# MAIN
file_name = f"{ALBUM}.json"

existing = {"photos": [], "clusters": []}

if os.path.exists(file_name):
    existing = json.load(open(file_name))

processed = set([p["id"] for p in existing["photos"]])
results = existing["photos"]

files = get_files()

for f in files:
    if f["id"] in processed:
        continue

    print("Nova:", f["name"])

    url = f"https://drive.google.com/thumbnail?id={f['id']}&sz=w800"

    emb = get_embedding(url)

    if not emb:
        continue

    results.append({
        "id": f["id"],
        "name": f["name"],
        "descriptor": emb
    })

clusters = cluster(results)

json.dump({
    "photos": results,
    "clusters": clusters
}, open(file_name, "w"))

print("Final:", len(results))

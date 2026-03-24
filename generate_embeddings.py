import os
import json
import requests
import numpy as np
import urllib.parse
import cv2
from PIL import Image
from io import BytesIO
from insightface.app import FaceAnalysis

# =============================
# CONFIG
# =============================

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
    raise Exception(f"Álbum inválido: {ALBUM}")

print("📁 Álbum:", ALBUM)

# =============================
# MODELO
# =============================

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# =============================
# HELPERS
# =============================

def normalize(v):
    return (v / np.linalg.norm(v)).tolist()

def get_drive_files():
    query = f"'{FOLDER_ID}' in parents and mimeType contains 'image/' and trashed=false"
    encoded_query = urllib.parse.quote(query)

    url = f"https://www.googleapis.com/drive/v3/files?q={encoded_query}&fields=files(id,name)&pageSize=1000&key={API_KEY}"

    res = requests.get(url).json()

    if "error" in res:
        print("❌ ERRO API:", res)
        return []

    files = res.get("files", [])
    print(f"📸 Total encontrados: {len(files)}")

    return files

def get_embeddings(url):
    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return []

        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        faces = app.get(img)

        if not faces:
            return []

        embeddings = []

        for face in faces:
            emb = face.embedding

            # 🔥 FLIP (melhora MUITO)
            img_flip = np.fliplr(img)
            faces_flip = app.get(img_flip)

            if faces_flip:
                emb_flip = faces_flip[0].embedding
                emb = (emb + emb_flip) / 2

            embeddings.append(normalize(emb))

        return embeddings

    except Exception as e:
        print("⚠️ Erro:", e)
        return []

def cluster_faces(data):
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

    print(f"👥 Clusters: {len(groups)}")
    return groups

# =============================
# MAIN
# =============================

file_name = f"{ALBUM}_py.json"

existing = {"photos": [], "clusters": []}

if os.path.exists(file_name):
    existing = json.load(open(file_name))

processed = set(p["id"] for p in existing["photos"])
results = existing["photos"]

files = get_drive_files()

count = 0

for f in files:
    if f["id"] in processed:
        continue

    print("🆕", f["name"])

    url = f"https://drive.google.com/thumbnail?id={f['id']}&sz=w800"

    embs = get_embeddings(url)

    if not embs:
        continue

    for emb in embs:
        results.append({
            "id": f["id"],
            "name": f["name"],
            "descriptor": emb
        })

    count += 1

print("✅ Novas:", count)
print("📊 Total:", len(results))

clusters = cluster_faces(results)

json.dump({
    "photos": results,
    "clusters": clusters
}, open(file_name, "w"))

print("💾 Salvo:", file_name)

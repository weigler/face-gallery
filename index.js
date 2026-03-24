const faceapi = require('face-api.js');
const canvas = require('canvas');
const fetch = require('node-fetch');
const fs = require('fs');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData, fetch });

const API_KEY = process.env.API_KEY;
const ALBUM = process.env.ALBUM;

const ALBUMS = {
  RPM2406: "1-LyABC7nFLJ9M1j3k1iZHT0LxAYoqje2",
  RPM2407: "1-It0lAedNjHY4lS0nUU1EXxIgpLn3L_Z",
  RPM2410: "1-D_FwO4HKzbWITv3ZRnVpozLe2LUGEh8",
  RPM2411: "1-yPDxcJkwYxJcTCWEC3chbCQF-eFMv0V",
  RPM2412: "1-xWf_2z6xqXqREnqsTQ0JQfndD-8TW5D",
  RPM2501: "1-TL7LLUmmbEfrTvZDUp7zC7gDuf7TOrv",
  RPM2502: "1FL2x8eeN_2tTHRFEi7f4oetHSWvKnhNd",
  RPM2503: "1FLAHTI74-PhoreWcQcfDtcLtDYLiD-2J",
  RPM2504: "1iu0SS8JCpAmpb-MXfrJH8Jj7qHdMwrOe",
  RPM2505: "1HREu2ddVkgfRcWP2SwobLzuf4xq4uaAs",
  RPM2506: "10btbePXG2PYEkz5rYxKX2sOyjCSNRjuL",
  RPM2601: "1DtvBHNkBXS7FyzkF_lutZrerY6S7rDkF",
  GISELA60: "1Wz5I-D-K7wLwP9ExMUdawXkkAK-i2KXT",
  ALTORIO26: "1hMHJ9kKDb-PPwcOYOfqNwaAgKMd54lMT",
};

const FOLDER_ID = ALBUMS[ALBUM];

if (!FOLDER_ID) {
  throw new Error("Álbum inválido: " + ALBUM);
}

const MODEL_PATH = './models';

// carregar modelos
async function loadModels() {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
}

// buscar arquivos
async function getDriveFiles() {
  const query = encodeURIComponent(
    `'${FOLDER_ID}' in parents and mimeType contains 'image/' and trashed=false`
  );

  const url = `https://www.googleapis.com/drive/v3/files?q=${query}&fields=files(id,name)&pageSize=1000&key=${API_KEY}`;

  const res = await fetch(url);
  const data = await res.json();

  // 🔥 DEBUG CORRETO (aqui sim funciona)
  if (!data.files) {
    console.log("Erro API:", data);
  }

  return data.files || [];
}

// detectar rosto
async function getDescriptors(url) {
  try {
    const img = await canvas.loadImage(url);

    const c = canvas.createCanvas(512, 512);
    const ctx = c.getContext('2d');

    const size = Math.min(img.width, img.height);

    ctx.drawImage(
      img,
      (img.width - size) / 2,
      (img.height - size) / 2,
      size,
      size,
      0,
      0,
      512,
      512
    );

    const detections = await faceapi
      .detectAllFaces(c)
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (!detections.length) return [];

    return detections.map(d => Array.from(d.descriptor));

  } catch {
    return [];
  }
}

// 🔥 CLUSTER SIMPLES (agrupamento)
function clusterFaces(data) {
  const groups = [];

  data.forEach(item => {
    let added = false;

    for (let group of groups) {
      const dist = faceapi.euclideanDistance(
        item.descriptor,
        group[0].descriptor
      );

      if (dist < 0.45) {
        group.push(item);
        added = true;
        break;
      }
    }

    if (!added) groups.push([item]);
  });

  return groups;
}

// MAIN
(async () => {
  await loadModels();

  const files = await getDriveFiles();

  const fileName = `${ALBUM}.json`;
  
  let existingData = { photos: [], clusters: [] };
  
  if (fs.existsSync(fileName)) {
  const raw = JSON.parse(fs.readFileSync(fileName));

  // 🔥 compatibilidade com formato antigo
  if (Array.isArray(raw)) {
    existingData.photos = raw;
  } else {
    existingData = raw;
  }
}

  const processed = new Set(existingData.photos.map(f => f.id));
  const results = [...existingData.photos];

  for (let file of files) {

    if (processed.has(file.id)) continue;

    console.log("Nova:", file.name);

    const url = `https://drive.google.com/thumbnail?id=${file.id}&sz=w800`;

    const descriptors = await getDescriptors(url);

       if (!descriptors.length) continue;

      for (const descriptor of descriptors) {
        results.push({
          id: file.id,
          name: file.name,
          descriptor
      });
}

  // 🔥 gerar clusters
  const clusters = clusterFaces(results);

  fs.writeFileSync(fileName, JSON.stringify({
    photos: results,
    clusters: clusters
  }));

  console.log("Final:", results.length);

})();

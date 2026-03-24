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
  GISELA60: "1Wz5I-D-K7wLwP9ExMUdawXkkAK-i2KXT",
  ALTORIO26: "1hMHJ9kKDb-PPwcOYOfqNwaAgKMd54lMT",
};

const FOLDER_ID = ALBUMS[ALBUM];
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
  return data.files || [];
}

// detectar rosto
async function getDescriptor(url) {
  try {
    const img = await canvas.loadImage(url);

    const c = canvas.createCanvas(512, 512);
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, 512, 512);

    const det = await faceapi
      .detectSingleFace(c)
      .withFaceLandmarks()
      .withFaceDescriptor();

    return det ? Array.from(det.descriptor) : null;

  } catch {
    return null;
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

      if (dist < 0.5) {
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

  let existing = [];
  const fileName = `${ALBUM}.json`;

  if (fs.existsSync(fileName)) {
    existing = JSON.parse(fs.readFileSync(fileName));
  }

  const processed = new Set(existing.map(f => f.id));
  const results = [...existing];

  for (let file of files) {

    if (processed.has(file.id)) continue;

    console.log("Nova:", file.name);

    const url = `https://drive.google.com/thumbnail?id=${file.id}&sz=w800`;

    const descriptor = await getDescriptor(url);

    if (!descriptor) continue;

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

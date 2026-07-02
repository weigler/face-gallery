// NOTA: @tensorflow/tfjs-node NÃO é usado aqui de propósito.
// face-api.js@0.22.2 embute uma versão antiga do @tensorflow/tfjs-core.
// Carregar o tfjs-node (versão 4.x) registra um tfjs-core incompatível
// e quebra em runtime (kernel "forwardFunc_1 is not a function").
// O aviso "slow CPU backend" no log é inofensivo — só indica que
// está usando o backend puro em JS, que é mais lento mas funciona.
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
  GE2026: "1RQ0zqtWNVZljZN9xWZP1DeLXt5vC5Im1",
};

const FOLDER_ID = ALBUMS[ALBUM];

if (!FOLDER_ID) {
  throw new Error("Álbum inválido: " + ALBUM);
}

const MODEL_PATH = './models';

// ---------------------------------------------------------------------------
// Utilidades
// ---------------------------------------------------------------------------

function log(...args) {
  const ts = new Date().toISOString().substring(11, 19);
  console.log(`[${ts}]`, ...args);
}

function formatDuration(ms) {
  const s = Math.floor(ms / 1000);
  const min = Math.floor(s / 60);
  const sec = s % 60;
  return min > 0 ? `${min}m ${sec}s` : `${sec}s`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Modelos
// ---------------------------------------------------------------------------

async function loadModels() {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
}

// ---------------------------------------------------------------------------
// Google Drive: listagem com cache em memória + retry
// ---------------------------------------------------------------------------

// Evita reconsultar a mesma pasta mais de uma vez durante a execução
// (ex.: se um atalho do Drive apontar para uma pasta já visitada).
const driveCache = new Map();
let apiCalls = 0;

async function driveListChildren(folderId, retries = 3) {
  if (driveCache.has(folderId)) {
    return driveCache.get(folderId);
  }

  const query = encodeURIComponent(`'${folderId}' in parents and trashed=false`);
  const url = `https://www.googleapis.com/drive/v3/files?q=${query}&fields=files(id,name,mimeType)&pageSize=1000&key=${API_KEY}`;

  for (let attempt = 1; attempt <= retries; attempt++) {
    apiCalls++;
    try {
      const res = await fetch(url);
      const data = await res.json();

      if (data.error) {
        throw new Error(data.error.message || JSON.stringify(data.error));
      }

      const children = data.files || [];
      const folders = children.filter(
        (f) => f.mimeType === 'application/vnd.google-apps.folder'
      );
      const files = children.filter((f) => f.mimeType.startsWith('image/'));

      const result = { folders, files };
      driveCache.set(folderId, result);
      return result;
    } catch (err) {
      log(`⚠️  Erro ao listar pasta ${folderId} (tentativa ${attempt}/${retries}): ${err.message}`);
      if (attempt === retries) {
        log(`❌ Desistindo da pasta ${folderId} após ${retries} tentativas.`);
        return { folders: [], files: [] };
      }
      await sleep(1000 * attempt);
    }
  }
}

// Percorre recursivamente todas as subpastas a partir de uma pasta raiz
async function walkDrive(folderId, folderName, parentPath = []) {
  const currentPath = parentPath.concat(folderName).join('/');
  log(`📁 Lendo pasta: ${currentPath}`);

  const { folders, files } = await driveListChildren(folderId);
  log(`   → ${files.length} imagem(ns), ${folders.length} subpasta(s)`);

  let allFiles = files.map((f) => ({
    id: f.id,
    name: f.name,
    folder: currentPath,
  }));

  for (const sub of folders) {
    const subFiles = await walkDrive(sub.id, sub.name, parentPath.concat(folderName));
    allFiles = allFiles.concat(subFiles);
  }

  return allFiles;
}

// ---------------------------------------------------------------------------
// Detecção de rosto(s)
// ---------------------------------------------------------------------------

async function getDescriptors(url, fileName) {
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

    detections.sort(
      (a, b) =>
        b.detection.box.width * b.detection.box.height -
        a.detection.box.width * a.detection.box.height
    );

    if (!detections.length) {
      log(`   ⚠️  Nenhum rosto detectado em: ${fileName}`);
      return [];
    }

    return detections.map((d) => Array.from(d.descriptor));
  } catch (err) {
    log(`   ❌ Erro ao processar "${fileName}": ${err.message}`);
    return [];
  }
}

// ---------------------------------------------------------------------------
// Clusterização simples
// ---------------------------------------------------------------------------

function clusterFaces(data) {
  const groups = [];

  data.forEach((item) => {
    let added = false;

    for (const group of groups) {
      const dist = faceapi.euclideanDistance(item.descriptor, group[0].descriptor);

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

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

(async () => {
  const startTime = Date.now();
  log(`🚀 Iniciando processamento do álbum "${ALBUM}"`);
  log(`API_KEY: ${API_KEY ? API_KEY.substring(0, 10) + '...' : 'NÃO DEFINIDA'}`);

  await loadModels();
  log('✅ Modelos carregados');

  const files = await walkDrive(FOLDER_ID, ALBUM);
  log(`📸 Total de imagens encontradas (todas as subpastas): ${files.length}`);
  log(`🌐 Chamadas à API do Drive: ${apiCalls}`);

  const fileName = `${ALBUM}.json`;

  let existingData = { photos: [], clusters: [] };

  if (fs.existsSync(fileName)) {
    const raw = JSON.parse(fs.readFileSync(fileName));

    // Compatibilidade com o formato antigo (array simples de fotos)
    if (Array.isArray(raw)) {
      existingData.photos = raw;
    } else {
      existingData = raw;
    }
  }

  const processed = new Set(existingData.photos.map((f) => f.id));
  const results = [...existingData.photos];

  let novas = 0;
  let ignoradas = 0;
  let comErro = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    if (processed.has(file.id)) {
      ignoradas++;
      continue;
    }

    log(`(${i + 1}/${files.length}) Nova imagem: "${file.name}" [${file.folder}]`);

    const url = `https://drive.google.com/thumbnail?id=${file.id}&sz=w800`;
    const descriptors = await getDescriptors(url, file.name);

    if (!descriptors.length) {
      comErro++;
      continue;
    }

    for (const descriptor of descriptors) {
      results.push({
        id: file.id,
        name: file.name,
        folder: file.folder,
        descriptor,
      });
    }

    novas++;
  }

  log('🔗 Gerando clusters...');
  const clusters = clusterFaces(results);

  fs.writeFileSync(
    fileName,
    JSON.stringify({
      photos: results,
      clusters: clusters,
    })
  );

  const elapsed = formatDuration(Date.now() - startTime);

  log('========================================');
  log('              RESUMO FINAL              ');
  log('========================================');
  log(`Álbum:                     ${ALBUM}`);
  log(`Imagens encontradas:       ${files.length}`);
  log(`Fotos novas processadas:   ${novas}`);
  log(`Fotos já existentes:       ${ignoradas}`);
  log(`Fotos sem rosto/com erro:  ${comErro}`);
  log(`Total de descritores:      ${results.length}`);
  log(`Clusters gerados:          ${clusters.length}`);
  log(`Chamadas à API do Drive:   ${apiCalls}`);
  log(`Tempo total:               ${elapsed}`);
  log('========================================');
})().catch((err) => {
  console.error('❌ Erro fatal:', err);
  process.exit(1);
});

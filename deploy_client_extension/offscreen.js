// offscreen.js – runs inside the hidden offscreen document
// Implements the same inference logic as test.py (scalar → room index → room name)

let session = null;
let roomNames = [];
let maxLength = 128;
let isModelReady = false;
let tokenizerStoi = null;
let padId = 0, unkId = 0, clsId = 1, sepId = 2; // special token ids (must match training)

// ------------------------------------------------------------------
// 1. Load tokenizer (character‑level) from tokenizer.json
// ------------------------------------------------------------------
async function loadTokenizer() {
  const resp = await fetch(chrome.runtime.getURL('tokenizer.json'));
  const json = await resp.json();
  const vocab = json.model.vocab;          // { "a": 0, "b": 1, ... }
  const stoi = new Map();
  for (const [ch, id] of Object.entries(vocab)) {
    stoi.set(ch, id);
  }
  // Special token ids (the JSON should contain them)
  padId = stoi.get('[PAD]') ?? 0;
  unkId = stoi.get('[UNK]') ?? 3;
  clsId = stoi.get('[CLS]') ?? 1;
  sepId = stoi.get('[SEP]') ?? 2;
  return stoi;
}

// ------------------------------------------------------------------
// 2. Load room names from rooms.json (generated from edges.tsv)
// ------------------------------------------------------------------
async function loadRooms() {
  const resp = await fetch(chrome.runtime.getURL('rooms.json'));
  return await resp.json();   // array of room names, same order as model's training
}

// ------------------------------------------------------------------
// 3. Tokenization (exactly as in test.py / evaluate_rows_into)
//    - character‑level
//    - add [CLS] at beginning, [SEP] at end
//    - truncate to maxLength - 2
//    - pad to maxLength with [PAD]
// ------------------------------------------------------------------
function tokenize(text, stoi, maxLen) {
  // Character tokenization
  const chars = text.split('');
  let ids = chars.map(ch => stoi.get(ch) ?? unkId);
  // Truncate to make room for [CLS] and [SEP]
  if (ids.length > maxLen - 2) ids = ids.slice(0, maxLen - 2);
  const inputIds = [clsId, ...ids, sepId];
  const attentionMask = new Array(inputIds.length).fill(1);
  // Pad
  while (inputIds.length < maxLen) {
    inputIds.push(padId);
    attentionMask.push(0);
  }
  return { inputIds, attentionMask };
}

// ------------------------------------------------------------------
// 4. Inference: scalar output → room index → room name
//    Matches evaluate_rows_into: model returns a scalar, we round to nearest integer
// ------------------------------------------------------------------
async function predictRoom(prompt) {
  if (!session || !tokenizerStoi || !isModelReady) {
    return "Model not ready yet.";
  }
  try {
    const { inputIds, attentionMask } = tokenize(prompt, tokenizerStoi, maxLength);
    const inputTensor = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(x => BigInt(x))), [1, maxLength]);
    const maskTensor = new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(x => BigInt(x))), [1, maxLength]);
    const feeds = { input_ids: inputTensor, attention_mask: maskTensor };
    const results = await session.run(feeds);
    // The model outputs a scalar (shape []) – this is the raw prediction
    const scalarValue = results.logits.data[0];   // output name 'logits'
    // Round to nearest integer to get room index (same as test.py)
    const roomIndex = Math.round(scalarValue);
    if (roomIndex >= 0 && roomIndex < roomNames.length) {
      return `You are heading to ${roomNames[roomIndex]}.`;
    } else {
      return `I'm not sure which room you mean. (Predicted index: ${roomIndex})`;
    }
  } catch (err) {
    console.error('Prediction error:', err);
    return `Error: ${err.message}`;
  }
}

// ------------------------------------------------------------------
// 5. Initialisation (load model, tokenizer, rooms in parallel)
// ------------------------------------------------------------------
async function init() {
  try {
    const modelPath = chrome.runtime.getURL('model.onnx');
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    });
    const [stoi, rooms] = await Promise.all([loadTokenizer(), loadRooms()]);
    tokenizerStoi = stoi;
    roomNames = rooms;
    isModelReady = true;
    console.log('Offscreen: model, tokenizer, and rooms ready (test.py compatible)');
  } catch (err) {
    console.error('Init error:', err);
    isModelReady = false;
  }
}
init();

// ------------------------------------------------------------------
// 6. Listen for chat requests from popup
// ------------------------------------------------------------------
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'CHAT_REQUEST') {
    if (!isModelReady) {
      sendResponse({ reply: "Model is still loading. Please wait a few seconds and try again." });
      return true;
    }
    predictRoom(request.prompt)
      .then(reply => sendResponse({ reply }))
      .catch(err => sendResponse({ reply: `Error: ${err.message}` }));
    return true;
  }
});
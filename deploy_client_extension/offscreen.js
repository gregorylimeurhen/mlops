// ort is already loaded and available as a global variable
let session = null;
let tokenizer = null;

// ---------- BERT Tokenizer (same as before) ----------
class BertTokenizer {
  constructor(tokenizerJson) {
    this.vocab = tokenizerJson.model.vocab;
    this.unkToken = '[UNK]';
    this.clsToken = '[CLS]';
    this.sepToken = '[SEP]';
    this.padToken = '[PAD]';
    this.maxLen = 128;
    this.vocabMap = new Map();
    for (const [token, id] of Object.entries(this.vocab)) {
      this.vocabMap.set(token, id);
    }
  }
  tokenize(text) {
    text = text.toLowerCase().trim();
    const words = text.split(/\s+/);
    const tokens = [];
    for (const word of words) {
      let start = 0;
      while (start < word.length) {
        let end = word.length;
        let found = false;
        while (end > start) {
          const sub = (start === 0 ? '' : '##') + word.substring(start, end);
          if (this.vocabMap.has(sub)) {
            tokens.push(sub);
            start = end;
            found = true;
            break;
          }
          end--;
        }
        if (!found) {
          tokens.push('[UNK]');
          break;
        }
      }
    }
    return tokens;
  }
  encode(text) {
    const tokens = this.tokenize(text);
    let truncated = tokens.slice(0, this.maxLen - 2);
    const inputIds = [
      this.vocabMap.get(this.clsToken),
      ...truncated.map(t => this.vocabMap.get(t) || this.vocabMap.get(this.unkToken)),
      this.vocabMap.get(this.sepToken)
    ];
    const attentionMask = new Array(inputIds.length).fill(1);
    const padId = this.vocabMap.get(this.padToken);
    while (inputIds.length < this.maxLen) {
      inputIds.push(padId);
      attentionMask.push(0);
    }
    return { inputIds, attentionMask };
  }
}

// Mean pooling function (same as before)
function meanPooling(tokenEmbeddings, attentionMask) {
  const batch = tokenEmbeddings.dims[0];
  const seqLen = tokenEmbeddings.dims[1];
  const hidden = tokenEmbeddings.dims[2];
  const embData = tokenEmbeddings.data;
  const maskData = attentionMask.data;
  const sumData = new Float32Array(batch * hidden);
  const countData = new Float32Array(batch * hidden);
  for (let i = 0; i < batch; i++) {
    for (let j = 0; j < seqLen; j++) {
      const maskVal = maskData[i * seqLen + j];
      if (maskVal === 0) continue;
      for (let k = 0; k < hidden; k++) {
        const idx = i * hidden + k;
        const embIdx = i * seqLen * hidden + j * hidden + k;
        sumData[idx] += embData[embIdx];
        countData[idx] += 1;
      }
    }
  }
  const pooled = new Float32Array(batch * hidden);
  for (let i = 0; i < batch * hidden; i++) {
    pooled[i] = sumData[i] / (countData[i] + 1e-9);
  }
  return new ort.Tensor('float32', pooled, [batch, hidden]);
}

// Load model and tokenizer
async function loadModelAndTokenizer() {
  try {
    const modelPath = chrome.runtime.getURL('model.onnx');
    
    // Configure ONNX Runtime to use only the WASM backend
    const sessionOptions = {
      executionProviders: ['wasm'],          // only use WebAssembly
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      wasm: {
        // Point to the exact WASM file you have (simd-threaded version)
        wasmPaths: chrome.runtime.getURL(''),
        // The library will append the correct filename; or you can set full path:
        // wasmUrl: chrome.runtime.getURL('ort-wasm-simd-threaded.wasm')
      }
    };

    session = await ort.InferenceSession.create(modelPath, sessionOptions);
    
    const tokenizerUrl = chrome.runtime.getURL('tokenizer.json');
    const response = await fetch(tokenizerUrl);
    const tokenizerJson = await response.json();
    tokenizer = new BertTokenizer(tokenizerJson);
    console.log('Offscreen: model & tokenizer ready');
  } catch (e) {
    console.error('Offscreen load error:', e);
  }
}
loadModelAndTokenizer();

// Inference function
async function getSentenceEmbedding(text) {
  if (!session || !tokenizer) return null;
  try {
    const { inputIds, attentionMask } = tokenizer.encode(text);
    const inputTensor = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(x => BigInt(x))), [1, tokenizer.maxLen]);
    const maskTensor = new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(x => BigInt(x))), [1, tokenizer.maxLen]);
    const feeds = { input_ids: inputTensor, attention_mask: maskTensor };
    const results = await session.run(feeds);
    const pooled = meanPooling(results.last_hidden_state, maskTensor);
    return Array.from(pooled.data);
  } catch (err) {
    console.error('Inference error:', err);
    return null;
  }
}

// Listen for messages from background
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'GET_EMBEDDING') {
    getSentenceEmbedding(request.text).then(emb => {
      sendResponse({ embedding: emb });
    }).catch(err => {
      sendResponse({ embedding: null });
    });
    return true;
  }
});
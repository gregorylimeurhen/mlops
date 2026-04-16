let activePopup = null;
let debounceTimer;

// ----- Predefined school locations (will be embedded later) -----
const LOCATIONS = ['Library', 'Cafeteria', 'Gym', 'Main Hall', 'Science Lab', 'Admin Office'];
let locationEmbeddings = null;  // will store embedding for each location

// ----- Cosine similarity -----
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ----- Get embedding for any text (via background/offscreen) -----
async function getEmbedding(text) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: 'GET_EMBEDDING', text }, (response) => {
      resolve(response?.embedding || null);
    });
  });
}

// ----- Precompute embeddings for all locations -----
async function precomputeLocationEmbeddings() {
  if (locationEmbeddings) return;
  const embeddings = [];
  for (const loc of LOCATIONS) {
    const emb = await getEmbedding(loc);
    embeddings.push(emb);
    console.log(`Embedding for "${loc}" ready`);
  }
  locationEmbeddings = embeddings;
}

// ----- Find best matching location -----
function findBestMatch(queryEmbedding) {
  let bestIdx = -1;
  let bestScore = -1;
  for (let i = 0; i < LOCATIONS.length; i++) {
    if (!locationEmbeddings[i]) continue;
    const score = cosineSimilarity(queryEmbedding, locationEmbeddings[i]);
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return bestScore > 0.5 ? LOCATIONS[bestIdx] : null;  // threshold 0.5
}

// ----- Show popup near the input -----
function showSuggestions(suggestions, targetInput) {
  removePopup();
  if (!suggestions || suggestions.length === 0) return;

  const popup = document.createElement('div');
  popup.className = 'venusaur-suggest';
  popup.style.cssText = `
    position: absolute;
    background: white;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-family: sans-serif;
    font-size: 14px;
    z-index: 10000;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
  `;

  suggestions.forEach(sug => {
    const item = document.createElement('div');
    item.textContent = sug;
    item.style.cssText = 'padding: 4px 8px; cursor: pointer;';
    item.onmouseover = () => item.style.backgroundColor = '#f0f0f0';
    item.onmouseout = () => item.style.backgroundColor = 'white';
    item.onclick = () => {
      targetInput.value = sug;
      targetInput.dispatchEvent(new Event('input', { bubbles: true }));
      removePopup();
    };
    popup.appendChild(item);
  });

  const rect = targetInput.getBoundingClientRect();
  popup.style.left = `${rect.left + window.scrollX}px`;
  popup.style.top = `${rect.bottom + window.scrollY + 4}px`;
  document.body.appendChild(popup);
  activePopup = popup;
}

function removePopup() {
  if (activePopup) {
    activePopup.remove();
    activePopup = null;
  }
}

// ----- Process user input -----
async function processInput(text) {
  if (text.length < 3) {
    removePopup();
    return;
  }
  const embedding = await getEmbedding(text);
  if (!embedding) return;
  if (!locationEmbeddings) await precomputeLocationEmbeddings();
  const bestLocation = findBestMatch(embedding);
  if (bestLocation) {
    showSuggestions([bestLocation], document.activeElement);
  } else {
    removePopup();
  }
}

// ----- Event listener with debounce -----
function onInput(event) {
  const input = event.target;
  if (!input.isContentEditable && input.tagName !== 'INPUT' && input.tagName !== 'TEXTAREA') return;
  const text = input.value || input.innerText;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => processInput(text), 600);
}

// ----- Inject listeners into all current and future inputs -----
function attachListeners() {
  document.querySelectorAll('input, textarea, [contenteditable="true"]').forEach(el => {
    if (!el.hasAttribute('data-venusaur')) {
      el.setAttribute('data-venusaur', 'true');
      el.addEventListener('input', onInput);
      el.addEventListener('blur', removePopup);
    }
  });
}
attachListeners();
new MutationObserver(attachListeners).observe(document.body, { childList: true, subtree: true });
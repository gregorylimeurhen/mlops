let offscreenReady = false;

async function ensureOffscreen() {
  if (offscreenReady) return;
  const existing = await chrome.runtime.getContexts({
    contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
    documentUrls: [chrome.runtime.getURL('offscreen.html')]
  });
  if (existing.length > 0) {
    offscreenReady = true;
    return;
  }
  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: [chrome.offscreen.Reason.DOM_PARSER],
    justification: 'Run ONNX Runtime model'
  });
  offscreenReady = true;
  console.log('Offscreen document created');
}

ensureOffscreen()

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'GET_EMBEDDING') {
    ensureOffscreen().then(() => {
      // forward to offscreen
      chrome.runtime.sendMessage(request).then(sendResponse).catch(err => sendResponse({ embedding: null, error: err.message }));
    });
    return true;
  }
});

// optional: log messages from offscreen
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.from === 'offscreen' && msg.type === 'log') {
    console.log('[Offscreen]', ...msg.args);
  }
});
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
    justification: 'Run ONNX model'
  });
  offscreenReady = true;
  console.log('Offscreen document created');
}

// Forward messages from popup to offscreen and back
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'CHAT_REQUEST') {
    ensureOffscreen().then(() => {
      chrome.runtime.sendMessage(request).then(response => {
        sendResponse(response);
      }).catch(err => sendResponse({ error: err.message }));
    });
    return true; // async
  }
});
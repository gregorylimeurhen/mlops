const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

function addMessage(text, sender) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
  msgDiv.textContent = text;
  messagesDiv.appendChild(msgDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendMessage() {
  const prompt = userInput.value.trim();
  if (!prompt) return;
  addMessage(prompt, 'user');
  userInput.value = '';
  addMessage('Thinking...', 'bot');
  
  try {
    const response = await chrome.runtime.sendMessage({
      type: 'CHAT_REQUEST',
      prompt: prompt
    });
    // Remove "Thinking..." and add actual reply
    messagesDiv.removeChild(messagesDiv.lastChild);
    if (response.reply) {
      addMessage(response.reply, 'bot');
    } else if (response.error) {
      addMessage(`Error: ${response.error}`, 'bot');
    } else {
      addMessage('No response from model', 'bot');
    }
  } catch (err) {
    messagesDiv.removeChild(messagesDiv.lastChild);
    addMessage(`Error: ${err.message}`, 'bot');
  }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') sendMessage();
});
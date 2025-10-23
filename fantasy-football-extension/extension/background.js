// background.js
const API_URL = 'https://your-api.herokuapp.com';  // Or your deployed API

// Fetch predictions on install
chrome.runtime.onInstalled.addListener(async () => {
  await fetchAndCachePredictions();
});

// Update predictions daily
chrome.alarms.create('updatePredictions', { periodInMinutes: 1440 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'updatePredictions') {
    fetchAndCachePredictions();
  }
});

async function fetchAndCachePredictions() {
  try {
    const currentWeek = getCurrentNFLWeek();
    const response = await fetch(
      `${API_URL}/predictions/weekly?week=${currentWeek}&season=2025`
    );
    
    if (!response.ok) throw new Error('API request failed');
    
    const predictions = await response.json();
    
    // Store in Chrome storage
    await chrome.storage.local.set({
      predictions: predictions,
      lastUpdated: Date.now(),
      week: currentWeek
    });
    
    console.log('Predictions updated:', Object.keys(predictions).length, 'players');
  } catch (error) {
    console.error('Failed to fetch predictions:', error);
  }
}

function getCurrentNFLWeek() {
  // Calculate current NFL week
  const seasonStart = new Date('2025-09-04');
  const now = new Date();
  const diff = now - seasonStart;
  const week = Math.floor(diff / (7 * 24 * 60 * 60 * 1000)) + 1;
  return Math.max(1, Math.min(18, week));
}

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getPrediction') {
    chrome.storage.local.get(['predictions'], (result) => {
      const prediction = result.predictions?.[request.playerName];
      sendResponse({ prediction });
    });
    return true;  // Async response
  }
});

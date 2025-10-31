const ext = typeof chrome !== 'undefined' ? chrome : (typeof browser !== 'undefined' ? browser : null);
let API_URL = 'http://localhost:5000';

// Fetch predictions on install and on browser startup
ext.runtime?.onInstalled.addListener(async () => {
  await ensurePredictionsFresh();
});
ext.runtime?.onStartup?.addListener(async () => {
  await ensurePredictionsFresh();
});

// Update predictions daily
ext.alarms?.create('updatePredictions', { periodInMinutes: 1440 });
ext.alarms?.onAlarm.addListener((alarm) => {
  if (alarm.name === 'updatePredictions') {
    fetchAndCachePredictions();
  }
});

async function fetchAndCachePredictions() {
  try {
    const currentWeek = getCurrentNFLWeek();
    const response = await fetch(`${API_URL}/predictions/weekly?week=${currentWeek}&season=2025`);
    
    if (!response.ok) throw new Error('API request failed');
    
  const predictions = await response.json();
    
    // Store in Chrome storage
    await ext.storage?.local.set({
      predictions: predictions,
      lastUpdated: Date.now(),
      week: currentWeek
    });
    
    console.log('Predictions updated:', Object.keys(predictions).length, 'players');
  } catch (error) {
    console.error('Failed to fetch predictions:', error);
  }
}

async function ensurePredictionsFresh(maxAgeMinutes = 360) { // default 6 hours
  try {
    const { lastUpdated } = await ext.storage?.local.get(['lastUpdated']);
    const stale = !lastUpdated || (Date.now() - lastUpdated) > maxAgeMinutes * 60 * 1000;
    if (stale) {
      await fetchAndCachePredictions();
    }
  } catch (e) {
    console.warn('Failed to check freshness, fetching anyway');
    await fetchAndCachePredictions();
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
ext.runtime?.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getPrediction') {
    ext.storage?.local.get(['predictions'], (result) => {
      const prediction = result.predictions?.[request.playerName];
      sendResponse({ prediction });
    });
    return true;  // Async response
  }
  if (request.action === 'ensurePredictions') {
    ensurePredictionsFresh().then(() => sendResponse({ ok: true })).catch(() => sendResponse({ ok: false }));
    return true;
  }
  if (request.action === 'refreshPredictions') {
    fetchAndCachePredictions().then(() => sendResponse({ ok: true })).catch(() => sendResponse({ ok: false }));
    return true;
  }
});

// content.js
let predictions = {};

// Load predictions from storage
chrome.storage.local.get(['predictions'], (result) => {
  predictions = result.predictions || {};
  console.log('Loaded', Object.keys(predictions).length, 'predictions');
  
  // Inject predictions
  injectPredictions();
  
  // Observer for dynamic content
  observePageChanges();
});

function injectPredictions() {
  // Platform-specific selectors
  const platformSelectors = detectPlatform();
  
  platformSelectors.forEach(selector => {
    const playerElements = document.querySelectorAll(selector);
    
    playerElements.forEach(element => {
      if (element.dataset.predictionInjected) return;
      
      const playerName = extractPlayerName(element);
      if (!playerName) return;
      
      const prediction = findPrediction(playerName);
      if (prediction) {
        injectPredictionBadge(element, prediction);
        element.dataset.predictionInjected = 'true';
      }
    });
  });
}

function detectPlatform() {
  const hostname = window.location.hostname;
  
  if (hostname.includes('yahoo')) {
    return [
      'a[data-ys-playerid]',
      '.ysf-player-name a',
      'td.player a'
    ];
  } else if (hostname.includes('espn')) {
    return [
      '.player-column__athlete',
      '.player-name a',
      '.playerinfo-column a'
    ];
  } else if (hostname.includes('sleeper')) {
    return [
      '.player-name',
      '[class*="PlayerCard"]'
    ];
  }
  
  return [];
}

function extractPlayerName(element) {
  let name = element.textContent.trim();
  
  // Remove position tags, team abbreviations
  name = name.replace(/\b(QB|RB|WR|TE|K|DEF)\b/gi, '').trim();
  name = name.replace(/\([A-Z]{2,3}\)/g, '').trim();
  name = name.replace(/\s+/g, ' ');
  
  return name;
}

function findPrediction(playerName) {
  // Exact match
  if (predictions[playerName]) {
    return predictions[playerName];
  }
  
  // Fuzzy match
  const keys = Object.keys(predictions);
  for (let key of keys) {
    // Handle "P. Mahomes" vs "Patrick Mahomes"
    const lastName = playerName.split(' ').pop();
    if (key.includes(lastName) && lastName.length > 3) {
      return predictions[key];
    }
  }
  
  return null;
}

function injectPredictionBadge(element, prediction) {
  // Create badge with prediction
  const badge = document.createElement('span');
  badge.className = 'ff-ai-prediction';
  badge.innerHTML = `
    <span class="ff-points">${prediction.predicted_points}</span>
    <span class="ff-label">AI</span>
  `;
  badge.title = `AI Prediction: ${prediction.predicted_points} pts (Transformer Model with NGS Data)`;
  
  // Position badge appropriately
  const container = element.closest('td') || element.closest('div') || element;
  container.style.position = 'relative';
  badge.style.position = 'absolute';
  badge.style.right = '5px';
  badge.style.top = '50%';
  badge.style.transform = 'translateY(-50%)';
  
  container.appendChild(badge);
}

function observePageChanges() {
  // Observe DOM mutations for SPAs
  const observer = new MutationObserver((mutations) => {
    // Debounce to avoid performance issues
    clearTimeout(window.predictionTimeout);
    window.predictionTimeout = setTimeout(() => {
      injectPredictions();
    }, 500);
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

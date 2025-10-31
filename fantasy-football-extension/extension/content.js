// Cross-browser API alias
const ext = typeof chrome !== 'undefined' ? chrome : (typeof browser !== 'undefined' ? browser : null);
let predictions = {};
let matchStats = { attempted: 0, injected: 0 };

// Ask background to ensure predictions are fresh (wakes MV3 service worker)
ext.runtime?.sendMessage({ action: 'ensurePredictions' }, () => {
  // Regardless of result, load from storage
  ext.storage?.local.get(['predictions', 'week', 'lastUpdated'], (result) => {
    predictions = result.predictions || {};
    console.log('Loaded', Object.keys(predictions).length, 'predictions');

    // Inject predictions
    injectPredictions();

    // Observer for dynamic content
    observePageChanges();

    // Create overlay with status/debug info
    createOrUpdateOverlay({
      loaded: Object.keys(predictions).length,
      attempted: matchStats.attempted,
      injected: matchStats.injected,
      week: result.week,
      lastUpdated: result.lastUpdated
    });
  });
});

function injectPredictions() {
  const platformSelectors = detectPlatform();

  platformSelectors.forEach(selector => {
    // Convert NodeList to array for filtering and slicing
    let playerElements = Array.from(document.querySelectorAll(selector));

    // Limit elements processed at once (adjust as needed)
    playerElements = playerElements.slice(0, 500);

    playerElements.forEach(element => {
      // Skip elements already injected or processed (including failed matches)
      if (element.dataset.predictionInjected || element.dataset.ffAiProcessed) return;

      // Skip invisible or detached elements
      if (!element.isConnected || element.offsetParent === null) {
        element.dataset.ffAiProcessed = 'true'; // mark processed to skip later
        return;
      }

      const playerName = extractPlayerName(element);
      if (!playerName) {
        element.dataset.ffAiProcessed = 'true';
        return;
      }
      
      matchStats.attempted++;
      const prediction = findPrediction(playerName);
      if (prediction) {
        injectPredictionBadge(element, prediction);
        element.dataset.predictionInjected = 'true';
        matchStats.injected++;
      } else {
        // Mark as processed to avoid repeated attempts on no-match
        element.dataset.ffAiProcessed = 'true';
        if (Math.random() < 0.01) {
          console.debug('[FF-AI] No prediction match for:', playerName);
        }
      }
    });
  });

  createOrUpdateOverlay({
    loaded: Object.keys(predictions).length,
    attempted: matchStats.attempted,
    injected: matchStats.injected
  });
}


function detectPlatform() {
  const hostname = window.location.hostname;
  
  if (hostname.includes('yahoo')) {
    return [
      'a[data-ys-playerid]',
      '.ysf-player-name a',
      'td.player a',
      'td.Nowrap a',
      'div.js-player-name a'
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
  // Remove injury/status markers
  name = name.replace(/\b(Q|O|D|DTD|IR|PUP|OUT|DNP|NA|P)\b/gi, '').trim();
  // Remove ranks or roster positions like RB1, WR2
  name = name.replace(/\b(RB|WR|TE|QB)\d+\b/gi, '').trim();
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
  let lastMutationCount = 0;
  
  const observer = new MutationObserver(mutations => {
    // Only trigger if number of mutations exceeds threshold to avoid trivial changes
    if (mutations.length < 5) return;

    clearTimeout(window.predictionTimeout);
    window.predictionTimeout = setTimeout(() => {
      injectPredictions();
    }, 1500); // Increased debounce to 1.5 seconds
  });

  observer.observe(document.body, { childList: true, subtree: true });
}


function createOrUpdateOverlay({ loaded, attempted, injected, week, lastUpdated } = {}) {
  let overlay = document.querySelector('.ff-ai-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.className = 'ff-ai-overlay';
    document.body.appendChild(overlay);
  }
  const parts = [];
  if (typeof loaded === 'number') parts.push(`Loaded: ${loaded}`);
  if (typeof injected === 'number' && typeof attempted === 'number') parts.push(`Injected: ${injected}/${attempted}`);
  if (typeof week !== 'undefined') parts.push(`Week: ${week}`);
  if (lastUpdated) {
    try {
      const dt = new Date(lastUpdated);
      parts.push(`Updated: ${dt.toLocaleTimeString()}`);
    } catch {}
  }
  overlay.textContent = `FF-AI ${parts.join(' | ')}`;
}

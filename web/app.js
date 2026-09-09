// Elements
const animeSearch = document.getElementById('anime-search');
const searchResults = document.getElementById('search-results');
const searchSpinner = document.getElementById('search-spinner');
const selectedPreview = document.getElementById('selected-anime-preview');
const clearBtn = document.getElementById('clear-selection');
const predictForm = document.getElementById('predict-form');
const malUsername = document.getElementById('mal-username');
const predictBtn = document.getElementById('predict-btn');
const btnText = document.querySelector('.btn-text');
const btnLoader = document.querySelector('.btn-loader');
const errorMessage = document.getElementById('error-message');

const dynamicBg = document.getElementById('dynamic-bg');
const emptyState = document.getElementById('empty-state');
const resultDisplay = document.getElementById('result-display');

// Loading overlay elements
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const loadingBarFill = document.getElementById('loading-bar-fill');
const previewLoadingBadge = document.getElementById('preview-loading-badge');

// State
let selectedAnime = null;
let searchTimeout = null;
let loadingMsgInterval = null;

// Loading overlay helpers
const loadingMessages = [
  'Analyzing viewing patterns...',
  'Fetching user history from MAL...',
  'Computing behavioral features...',
  'Running prediction model...',
  'Calculating drop probability...'
];

function showLoadingOverlay() {
  loadingOverlay.classList.remove('hidden');
  loadingBarFill.style.width = '0%';
  let msgIndex = 0;
  let progress = 0;
  
  loadingText.textContent = loadingMessages[0];
  
  loadingMsgInterval = setInterval(() => {
    msgIndex = (msgIndex + 1) % loadingMessages.length;
    loadingText.textContent = loadingMessages[msgIndex];
    progress = Math.min(progress + 18, 90);
    loadingBarFill.style.width = `${progress}%`;
  }, 1500);
}

function hideLoadingOverlay(success) {
  clearInterval(loadingMsgInterval);
  loadingBarFill.style.width = '100%';
  loadingText.textContent = success ? 'Analysis complete!' : 'Something went wrong.';
  
  setTimeout(() => {
    loadingOverlay.classList.add('hidden');
    loadingBarFill.style.width = '0%';
  }, 600);
}

// Debounced Search
animeSearch.addEventListener('input', (e) => {
  const query = e.target.value.trim();
  
  if (query.length < 3) {
    searchResults.classList.add('hidden');
    return;
  }

  clearTimeout(searchTimeout);
  searchSpinner.classList.remove('hidden');
  
  searchTimeout = setTimeout(async () => {
    try {
      const res = await fetch(`http://127.0.0.1:5000/search_anime?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      
      renderSearchResults(data.data);
    } catch (err) {
      console.error('Search failed', err);
    } finally {
      searchSpinner.classList.add('hidden');
    }
  }, 500);
});

// Hide search results when clicking outside
document.addEventListener('click', (e) => {
  if (!animeSearch.contains(e.target) && !searchResults.contains(e.target)) {
    searchResults.classList.add('hidden');
  }
});

// Generate a reliable base64-encoded SVG placeholder
function generatePlaceholder(title) {
  const initial = title ? title.charAt(0).toUpperCase() : '?';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="300">
    <defs>
      <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#ff5e3a"/>
        <stop offset="100%" stop-color="#ff9b44"/>
      </linearGradient>
    </defs>
    <rect width="200" height="300" fill="url(#g)"/>
    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
          font-family="sans-serif" font-size="100" fill="#ffffff" font-weight="bold">${initial}</text>
  </svg>`;
  return `data:image/svg+xml;base64,${btoa(svg)}`;
}

function renderSearchResults(results) {
  searchResults.innerHTML = '';
  
  if (!results || results.length === 0) {
    searchResults.innerHTML = '<li class="search-result-item"><span class="search-result-text">No results found</span></li>';
  } else {
    results.forEach(anime => {
      const li = document.createElement('li');
      li.className = 'search-result-item';
      
      const year = anime.year ? anime.year : 'Unknown Year';
      const placeholderUrl = generatePlaceholder(anime.title);
      const imgUrl = anime.images?.jpg?.small_image_url || placeholderUrl;
      
      li.innerHTML = `
        <img src="${imgUrl}" class="search-result-img" alt="" onerror="this.onerror=null;this.src='${placeholderUrl}'">
        <div class="search-result-text">
          <span class="search-result-title">${anime.title}</span>
          <span class="search-result-year">${year} • ${anime.episodes || '?'} eps</span>
        </div>
      `;
      
      li.addEventListener('click', () => selectAnime(anime));
      searchResults.appendChild(li);
    });
  }
  
  searchResults.classList.remove('hidden');
}

// Fetch rich data (poster, synopsis) from Jikan API
async function fetchJikanDetails(malId) {
  try {
    const res = await fetch(`https://api.jikan.moe/v4/anime/${malId}`);
    if (!res.ok) return null;
    const json = await res.json();
    return json.data || null;
  } catch (err) {
    console.warn('Jikan fetch failed, using local data only:', err);
    return null;
  }
}

async function selectAnime(anime) {
  selectedAnime = { ...anime };
  
  // Update UI immediately with what we have
  animeSearch.value = anime.title;
  searchResults.classList.add('hidden');
  
  const placeholder = generatePlaceholder(anime.title);
  const posterEl = document.getElementById('preview-poster');
  
  // Show preview immediately with placeholder
  posterEl.src = placeholder;
  posterEl.onerror = function() { this.onerror=null; this.src=placeholder; };
  posterEl.classList.add('poster-loading');
  document.getElementById('preview-title').textContent = anime.title;
  
  const year = anime.year ? anime.year : 'Unknown Year';
  document.getElementById('preview-year').textContent = year;
  document.getElementById('preview-episodes').textContent = anime.episodes || '?';
  
  selectedPreview.classList.remove('hidden');
  dynamicBg.style.backgroundImage = `linear-gradient(135deg, rgba(255, 94, 58, 0.2), rgba(20, 20, 25, 0.9))`;
  
  checkFormValidity();
  
  // Enrich with Jikan data in the background (poster + synopsis)
  if (anime.mal_id) {
    previewLoadingBadge.classList.remove('hidden');
    
    const jikan = await fetchJikanDetails(anime.mal_id);
    
    // Only update if user hasn't changed selection while we were fetching
    if (jikan && selectedAnime && selectedAnime.mal_id === anime.mal_id) {
      // Merge poster images
      if (jikan.images) {
        selectedAnime.images = jikan.images;
        const posterUrl = jikan.images.jpg?.large_image_url || jikan.images.jpg?.image_url || placeholder;
        posterEl.src = posterUrl;
        dynamicBg.style.backgroundImage = `url(${posterUrl})`;
      }
      
      // Merge synopsis
      if (jikan.synopsis) {
        selectedAnime.synopsis = jikan.synopsis;
      }
      
      // Merge themes if available
      if (jikan.themes) {
        selectedAnime.themes = jikan.themes;
      }
    }
    
    posterEl.classList.remove('poster-loading');
    previewLoadingBadge.classList.add('hidden');
  }
}

clearBtn.addEventListener('click', () => {
  selectedAnime = null;
  animeSearch.value = '';
  selectedPreview.classList.add('hidden');
  dynamicBg.style.backgroundImage = 'none';
  
  emptyState.classList.remove('hidden');
  resultDisplay.classList.add('hidden');
  
  checkFormValidity();
});

malUsername.addEventListener('input', checkFormValidity);

function checkFormValidity() {
  if (selectedAnime && malUsername.value.trim().length > 0) {
    predictBtn.disabled = false;
  } else {
    predictBtn.disabled = true;
  }
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorMessage.classList.remove('hidden');
}

function hideError() {
  errorMessage.classList.add('hidden');
}

// Prediction Submission
predictForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedAnime || !malUsername.value.trim()) return;

  // Loading state — show full overlay
  predictBtn.disabled = true;
  btnText.textContent = 'Analyzing...';
  btnLoader.classList.remove('hidden');
  hideError();
  showLoadingOverlay();

  try {
    // We already have stats directly inside selectedAnime from our local endpoint!
    const statsData = selectedAnime.stats || {};

    const response = await fetch('http://127.0.0.1:5000/predict_user_anime', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        anime_data: selectedAnime,
        stats_data: statsData,
        mal_username: malUsername.value.trim()
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed');
    }

    hideLoadingOverlay(true);
    // Small delay so user sees "Analysis complete!" before results appear
    await new Promise(r => setTimeout(r, 700));
    renderPrediction(data.predicted_drop_episode);

  } catch (err) {
    hideLoadingOverlay(false);
    showError(err.message);
  } finally {
    predictBtn.disabled = false;
    btnText.textContent = 'Run Prediction Analysis';
    btnLoader.classList.add('hidden');
  }
});

function renderPrediction(dropEp) {
  emptyState.classList.add('hidden');
  resultDisplay.classList.remove('hidden');
  
  const totalEps = selectedAnime.episodes || 12; // Fallback if unknown
  const placeholder = generatePlaceholder(selectedAnime.title);
  
  // Update details panel
  const detailImgUrl = selectedAnime.images?.jpg?.large_image_url || selectedAnime.images?.jpg?.image_url || placeholder;
  const detailPoster = document.getElementById('detail-poster');
  detailPoster.src = detailImgUrl;
  detailPoster.onerror = function() { this.onerror=null; this.src=placeholder; };
  
  document.getElementById('detail-title').textContent = selectedAnime.title;
  document.getElementById('detail-score-val').textContent = selectedAnime.score || 'N/A';
  document.getElementById('detail-synopsis-text').textContent = selectedAnime.synopsis || 'Synopsis not available.';
  
  const tagsContainer = document.getElementById('detail-tags');
  tagsContainer.innerHTML = '';
  const tags = [...(selectedAnime.genres || []), ...(selectedAnime.themes || [])].slice(0, 4);
  tags.forEach(tag => {
    const span = document.createElement('span');
    span.className = 'tag';
    span.textContent = tag.name;
    tagsContainer.appendChild(span);
  });
  
  // Animate the numbers
  animateNumber('pred-episode', dropEp, 1000);
  document.getElementById('total-episodes').textContent = totalEps;
  
  // Calculate and animate Risk Bar
  const dropRatio = dropEp / totalEps;
  const riskBar = document.getElementById('risk-bar');
  const riskLabel = document.getElementById('risk-label');
  
  // Reset bar width for animation
  riskBar.style.width = '0%';
  
  setTimeout(() => {
    // We show "completion percentage" meaning how far they get.
    // If they drop at ep 3 out of 12, that's 25% (High Risk of dropping early)
    const percentage = Math.min(100, Math.max(0, (dropEp / totalEps) * 100));
    riskBar.style.width = `${percentage}%`;
    
    if (percentage > 80) {
      riskBar.style.backgroundColor = 'var(--risk-low)';
      riskLabel.textContent = 'Low Risk: Likely to complete or drop very late.';
    } else if (percentage > 40) {
      riskBar.style.backgroundColor = 'var(--risk-med)';
      riskLabel.textContent = 'Medium Risk: May drop midway through.';
    } else {
      riskBar.style.backgroundColor = 'var(--risk-high)';
      riskLabel.textContent = 'High Risk: Likely to drop early.';
    }
  }, 100);
}

function animateNumber(elementId, target, duration) {
  const element = document.getElementById(elementId);
  const isFloat = !Number.isInteger(target);
  const start = 0;
  const startTime = performance.now();
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // Easing out cubic
    const easeProgress = 1 - Math.pow(1 - progress, 3);
    const current = start + (target - start) * easeProgress;
    
    element.textContent = isFloat ? current.toFixed(1) : Math.round(current);
    
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = isFloat ? target.toFixed(1) : target;
    }
  }
  
  requestAnimationFrame(update);
}

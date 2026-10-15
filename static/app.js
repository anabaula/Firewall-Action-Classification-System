const HISTORY_KEY = 'firewallai_prediction_history_v2';
const PAGE_TITLES = { dashboard: 'Dashboard', classify: 'Classify traffic', history: 'Prediction history' };
const ACTIONS = ['Allow', 'Deny', 'Drop', 'Reset'];

function getHistory() {
  try { const value = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); return Array.isArray(value) ? value : []; }
  catch { return []; }
}
function saveHistory(items) { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 100))); }
function esc(value) { return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function actionClass(action) { return ({Allow:'allow',Deny:'deny',Drop:'drop',Reset:'reset'})[action] || 'allow'; }
function actionPill(action) { return `<span class="action-pill action-${actionClass(action)}">${esc(action)}</span>`; }
function formatTime(iso) { const date = new Date(iso); return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString([], {month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'}); }
function confidenceText(value) { return value === null || value === undefined ? '—' : `${Number(value).toFixed(1)}%`; }

function showPage(name) {
  document.querySelectorAll('.page').forEach(page => page.classList.toggle('active', page.id === `page-${name}`));
  document.querySelectorAll('.nav-item').forEach(item => item.classList.toggle('active', item.dataset.page === name));
  document.getElementById('breadcrumb-current').textContent = PAGE_TITLES[name] || 'Dashboard';
  document.getElementById('sidebar').classList.remove('open');
  window.scrollTo({top: 0, behavior: 'smooth'});
  if (name === 'history') renderHistory();
}

document.querySelectorAll('[data-page]').forEach(button => button.addEventListener('click', () => showPage(button.dataset.page)));
document.querySelectorAll('[data-go]').forEach(button => button.addEventListener('click', () => showPage(button.dataset.go)));
document.getElementById('mobile-menu').addEventListener('click', () => document.getElementById('sidebar').classList.toggle('open'));
document.addEventListener('click', event => {
  const sidebar = document.getElementById('sidebar');
  if (window.innerWidth <= 900 && sidebar.classList.contains('open') && !sidebar.contains(event.target) && !event.target.closest('#mobile-menu')) sidebar.classList.remove('open');
});

async function checkModelStatus() {
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    const ready = health.model_ready;
    document.getElementById('side-status').textContent = ready ? 'Model ready' : 'Model not trained';
    document.getElementById('header-status').textContent = ready ? 'MODEL READY' : 'MODEL NOT TRAINED';
    document.querySelector('.system-pill').classList.toggle('warning', !ready);
  } catch {
    document.getElementById('side-status').textContent = 'API unavailable';
    document.getElementById('header-status').textContent = 'API UNAVAILABLE';
    document.querySelector('.system-pill').classList.add('warning');
  }
}

function renderStats() {
  const history = getHistory();
  document.getElementById('stat-total').textContent = history.length;
  document.getElementById('stat-allow').textContent = history.filter(row => row.action === 'Allow').length;
  document.getElementById('stat-deny').textContent = history.filter(row => row.action === 'Deny').length;
  document.getElementById('stat-other').textContent = history.filter(row => ['Drop','Reset'].includes(row.action)).length;
  const recent = history.slice(0, 5);
  const tbody = document.getElementById('recent-table');
  if (!recent.length) {
    tbody.innerHTML = '<tr><td class="empty-cell" colspan="6"><div class="empty-icon">⌁</div><b>No predictions yet</b><span>Run your first classification to see results here.</span><button class="text-link" data-go="classify">Start classifying →</button></td></tr>';
    tbody.querySelector('[data-go]')?.addEventListener('click', () => showPage('classify'));
    return;
  }
  tbody.innerHTML = recent.map(row => `<tr><td>${esc(formatTime(row.time))}</td><td>${esc(row.source_port)}</td><td>${esc(row.destination_port)}</td><td>${esc(row.total_packets)}</td><td>${actionPill(row.action)}</td><td class="confidence">${esc(confidenceText(row.confidence))}</td></tr>`).join('');
}

function renderHistory() {
  const search = document.getElementById('history-search').value.trim().toLowerCase();
  const filter = document.getElementById('history-filter').value;
  const all = getHistory();
  const rows = all.filter(row => {
    const matchesSearch = String(row.source_port).includes(search) || String(row.destination_port).includes(search);
    return matchesSearch && (filter === 'all' || row.action === filter);
  });
  const tbody = document.getElementById('history-table');
  if (!rows.length) {
    tbody.innerHTML = `<tr><td class="empty-cell" colspan="7">${all.length ? 'No records match your search.' : 'No history saved yet. Run a classification first.'}</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(row => `<tr><td>${esc(formatTime(row.time))}</td><td>${esc(row.source_port)}</td><td>${esc(row.destination_port)}</td><td>${esc(row.total_packets)}</td><td>${esc(row.nat_source_port)}</td><td>${actionPill(row.action)}</td><td class="confidence">${esc(confidenceText(row.confidence))}</td></tr>`).join('');
}

document.getElementById('history-search').addEventListener('input', renderHistory);
document.getElementById('history-filter').addEventListener('change', renderHistory);
document.getElementById('clear-history').addEventListener('click', () => {
  if (!getHistory().length) return;
  if (window.confirm('Clear all prediction history stored in this browser?')) {
    saveHistory([]); renderStats(); renderHistory();
  }
});

document.getElementById('traffic-form').addEventListener('reset', () => {
  document.getElementById('form-message').textContent = '';
  document.getElementById('form-message').className = 'form-message';
});

document.getElementById('traffic-form').addEventListener('submit', async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const message = document.getElementById('form-message');
  const button = document.getElementById('predict-btn');
  const payload = Object.fromEntries(new FormData(form).entries());
  for (const key of ['source_port','destination_port','nat_source_port','nat_destination_port','total_bytes','bytes_sent','bytes_received','total_packets','connection_duration','forward_packets','backward_packets']) payload[key] = Number(payload[key]);
  message.className = 'form-message';
  message.textContent = '';
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Classifying…';
  try {
    const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Prediction failed.');
    document.getElementById('result-badge').className = `action-badge ${actionClass(result.action)}`;
    document.getElementById('result-badge').textContent = result.action.toUpperCase();
    document.getElementById('result-title').textContent = ({Allow:'Traffic may be permitted',Deny:'Traffic may be rejected',Drop:'Traffic may be discarded',Reset:'Connection may be reset'})[result.action] || 'Prediction complete';
    document.getElementById('result-description').textContent = result.description;
    document.getElementById('result-confidence').textContent = confidenceText(result.confidence);
    document.getElementById('confidence-bar').style.width = `${Math.max(0, Math.min(100, Number(result.confidence || 0)))}%`;
    document.getElementById('result-panel').classList.remove('hidden');
    const record = {time:new Date().toISOString(), source_port:payload.source_port, destination_port:payload.destination_port, nat_source_port:payload.nat_source_port, total_packets:payload.total_packets, action:result.action, confidence:result.confidence};
    saveHistory([record, ...getHistory()]);
    renderStats();
    message.className = 'form-message success';
    message.textContent = 'Classification completed. Results saved in this browser.';
    document.getElementById('result-panel').scrollIntoView({behavior:'smooth', block:'start'});
  } catch (error) {
    message.className = 'form-message';
    message.textContent = error.message || 'Could not connect to the Flask API. Check that the server is running.';
  } finally {
    button.disabled = false;
    button.innerHTML = 'Predict firewall action <span class="arrow">→</span>';
  }
});

document.getElementById('classify-another').addEventListener('click', () => {
  document.getElementById('result-panel').classList.add('hidden');
  document.getElementById('traffic-form').scrollIntoView({behavior:'smooth',block:'start'});
});

checkModelStatus();
renderStats();

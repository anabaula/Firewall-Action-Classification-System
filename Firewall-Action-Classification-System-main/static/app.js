// NetSentry single-page behavior: check model, classify traffic, save history.
// Keep the existing browser storage key so old predictions remain available.
const HISTORY_KEY = 'firewallai_prediction_history_v2';
const HISTORY_PAGE_SIZE = 3;
let historyPage = 0;

const form = document.getElementById('traffic-form');
const message = document.getElementById('form-message');
const predictButton = document.getElementById('predict-btn');
const generateButton = document.getElementById('generate-btn');
const themeToggle = document.getElementById('theme-toggle');

themeToggle.addEventListener('click', () => {
  const isDark = document.documentElement.dataset.theme !== 'dark';
  document.documentElement.dataset.theme = isDark ? 'dark' : 'light';
  themeToggle.setAttribute('aria-pressed', String(isDark));
  themeToggle.setAttribute('aria-label', `Switch to ${isDark ? 'light' : 'dark'} mode`);
});

function escapeHTML(value) {
  const entities = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  return String(value ?? '').replace(/[&<>"']/g, character => entities[character]);
}

function confidenceText(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  // Display exactly two decimals without rounding the original model confidence up.
  return `${(Math.trunc(Number(value) * 100) / 100).toFixed(2)}%`;
}

function formatTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString([], {
    month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit'
  });
}

// Prediction history is stored in this browser, not in a server database.
function getHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

function saveHistory(records) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(records.slice(0, 100)));
}

function renderHistory() {
  const search = document.getElementById('history-search').value.trim().toLowerCase();
  const filter = document.getElementById('history-filter').value;
  const history = getHistory();

  // Keep the original history index so Details opens the correct record.
  const rows = history.map((record, index) => ({ record, index })).filter(({ record }) => {
    const matchesSearch =
      formatTime(record.time).toLowerCase().includes(search) ||
      String(record.action ?? '').toLowerCase().includes(search);
    return matchesSearch && (filter === 'all' || record.action === filter);
  });

  const tableBody = document.getElementById('history-table');
  if (rows.length === 0) {
    const label = history.length ? 'No matches.' : 'No predictions yet.';
    tableBody.innerHTML = `<tr><td colspan="4" class="empty-cell">${label}</td></tr>`;
    historyPage = 0;
    document.getElementById('history-page-info').textContent = '0 predictions';
    document.getElementById('history-prev').disabled = true;
    document.getElementById('history-next').disabled = true;
    return;
  }

  const pages = Math.ceil(rows.length / HISTORY_PAGE_SIZE);
  historyPage = Math.min(historyPage, pages - 1);
  const visible = rows.slice(historyPage * HISTORY_PAGE_SIZE, (historyPage + 1) * HISTORY_PAGE_SIZE);
  document.getElementById('history-page-info').textContent =
    `${rows.length} prediction${rows.length === 1 ? '' : 's'} · ${historyPage + 1} / ${pages}`;
  document.getElementById('history-prev').disabled = historyPage === 0;
  document.getElementById('history-next').disabled = historyPage === pages - 1;

  tableBody.innerHTML = visible.map(({ record, index }) => `
    <tr>
      <td>${escapeHTML(formatTime(record.time))}</td>
      <td><span class="history-action">${escapeHTML(record.action)}</span></td>
      <td>${escapeHTML(confidenceText(record.confidence))}</td>
      <td><button type="button" class="details-button" data-history-index="${index}">Details</button></td>
    </tr>
  `).join('');
}

// Show every input saved for a prediction. Old records may only have a few fields.
const detailsDialog = document.getElementById('history-details');
const inputFields = Array.from(form.querySelectorAll('input[name]')).map(input => ({
  name: input.name,
  label: input.labels?.[0]?.textContent.trim() || input.name
}));

document.getElementById('history-table').addEventListener('click', event => {
  const button = event.target.closest('[data-history-index]');
  if (!button) return;

  const record = getHistory()[Number(button.dataset.historyIndex)];
  if (!record) return;

  document.getElementById('details-summary').textContent =
    `${formatTime(record.time)} · ${record.action ?? 'Unknown'} · ${confidenceText(record.confidence)}`;

  const fields = document.getElementById('details-fields');
  fields.replaceChildren();
  for (const field of inputFields) {
    const wrapper = document.createElement('div');
    const label = document.createElement('dt');
    const value = document.createElement('dd');
    label.textContent = field.label;
    value.textContent = record[field.name] == null ? 'Not recorded' : String(record[field.name]);
    wrapper.append(label, value);
    fields.append(wrapper);
  }

  detailsDialog.showModal();
});

document.getElementById('details-close').addEventListener('click', () => detailsDialog.close());
document.getElementById('details-done').addEventListener('click', () => detailsDialog.close());

document.getElementById('history-search').addEventListener('input', () => {
  historyPage = 0;
  renderHistory();
});
document.getElementById('history-filter').addEventListener('change', () => {
  historyPage = 0;
  renderHistory();
});
document.getElementById('history-prev').addEventListener('click', () => {
  historyPage = Math.max(0, historyPage - 1);
  renderHistory();
});
document.getElementById('history-next').addEventListener('click', () => {
  historyPage += 1;
  renderHistory();
});
document.getElementById('clear-history').addEventListener('click', () => {
  if (getHistory().length && window.confirm('Clear the prediction history saved in this browser?')) {
    saveHistory([]);
    renderHistory();
  }
});

// Show the custom number arrows only on the input the user has focused.
function updateStepper(input) {
  const stepper = input.closest('.number-control').querySelector('.number-stepper');
  const current = input.valueAsNumber;
  const minimum = Number(input.min);
  const maximum = input.max ? Number(input.max) : Infinity;

  stepper.querySelector('[data-step="up"]').disabled =
    input.disabled || (Number.isFinite(current) && current >= maximum);
  stepper.querySelector('[data-step="down"]').disabled =
    input.disabled || (Number.isFinite(current) && current <= minimum);
}

function changeNumber(input, direction) {
  if (input.disabled || input.readOnly) return;

  if (input.step === 'any') {
    const current = Number.isFinite(input.valueAsNumber) ? input.valueAsNumber : 0;
    const next = Math.round((current + direction * 0.1) * 1e6) / 1e6;
    input.value = String(Math.max(Number(input.min), next));
  } else if (direction > 0) {
    input.stepUp();
  } else {
    input.stepDown();
  }

  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.focus({ preventScroll: true });
}

document.querySelectorAll('.number-control').forEach(control => {
  const input = control.querySelector('input[type="number"]');
  control.querySelectorAll('[data-step]').forEach(button => {
    button.addEventListener('click', () => {
      changeNumber(input, button.dataset.step === 'up' ? 1 : -1);
    });
  });
  input.addEventListener('input', () => updateStepper(input));
  updateStepper(input);
});

function showPrediction(result) {
  document.getElementById('result-empty').classList.add('hidden');
  document.getElementById('result-output').classList.remove('hidden');

  const actionLabel = document.getElementById('result-action');
  actionLabel.textContent = result.action.toUpperCase();
  actionLabel.dataset.action = result.action.toLowerCase();
  document.getElementById('result-confidence').textContent = confidenceText(result.confidence);
}

function clearPrediction() {
  document.getElementById('result-action').removeAttribute('data-action');
  document.getElementById('result-output').classList.add('hidden');
  document.getElementById('result-empty').classList.remove('hidden');
}

// Fill the form from a real cleaned UCI dataset record; never auto-classify.
generateButton.addEventListener('click', async () => {
  predictButton.disabled = true;
  generateButton.disabled = true;
  generateButton.textContent = 'Generating…';
  message.textContent = '';
  message.className = 'form-message';

  try {
    const response = await fetch('/api/sample', { cache: 'no-store' });
    const example = await response.json();
    if (!response.ok) throw new Error(example.error || 'Could not load a dataset example.');
    if (!example.features || inputFields.some(field =>
      !Number.isFinite(Number(example.features[field.name])) || example.features[field.name] == null
    )) throw new Error('The dataset example is incomplete.');

    for (const input of form.querySelectorAll('input[name]')) {
      input.value = String(example.features[input.name]);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    clearPrediction();
    message.textContent = '';
    message.className = 'form-message';
  } catch (error) {
    message.textContent = error.message || 'Could not load a dataset example.';
  } finally {
    predictButton.disabled = false;
    generateButton.disabled = false;
    generateButton.textContent = 'Generate';
  }
});

// Submit the 11 numeric fields to Flask, then show the result and save history.
form.addEventListener('submit', async event => {
  event.preventDefault();

  const values = Object.fromEntries(new FormData(form).entries());
  for (const field of Object.keys(values)) {
    values[field] = Number(values[field]);
  }

  message.textContent = '';
  message.className = 'form-message';
  predictButton.disabled = true;
  generateButton.disabled = true;
  predictButton.textContent = 'Classifying…';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(values)
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Could not classify this traffic record.');
    }

    showPrediction(result);
    // Save all 11 form inputs for the History > Details window.
    const record = {
      ...values,
      time: new Date().toISOString(),
      action: result.action,
      confidence: result.confidence
    };
    try {
    saveHistory([record, ...getHistory()]);
    historyPage = 0;
    renderHistory();
   } catch {
    // Do not display informational success messages.
  }
   message.textContent = '';
   message.className = 'form-message';
  } catch (error) {
    message.textContent = error.message || 'Could not reach the prediction server.';
  } finally {
    predictButton.disabled = false;
    generateButton.disabled = false;
    predictButton.textContent = 'Classify';
  }
});

form.addEventListener('reset', () => {
  message.textContent = '';
  message.className = 'form-message';
  clearPrediction();
  requestAnimationFrame(() => {
    form.querySelectorAll('input[type="number"]').forEach(updateStepper);
  });
});

renderHistory();

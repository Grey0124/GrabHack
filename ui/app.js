async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res.text();
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function renderTrace(state) {
  const trace = document.getElementById('trace');
  const summary = document.getElementById('summary');
  trace.innerHTML = '';
  if (summary) summary.innerHTML = '';
  const steps = state?.scratchpad || [];

  let lastAction = null;
  let lastReflection = null;
  let latestSuggestion = null;
  let latestInstruction = null;

  steps.forEach((entry) => {
    // Render each known key if present to avoid hiding reflection
    ['thought', 'action', 'observation', 'reflection'].forEach((key) => {
      if (entry[key] === undefined) return;
      const val = entry[key];
      const card = el('div', `step ${key}`);
      card.appendChild(el('div', 'step-title', key.toUpperCase()));
      const pre = el('pre');
      pre.textContent = typeof val === 'string' ? val : JSON.stringify(val, null, 2);
      card.appendChild(pre);
      trace.appendChild(card);

      if (key === 'action') lastAction = val;
      if (key === 'reflection') lastReflection = val;
      if (key === 'observation' && val && typeof val === 'object') {
        if (val.suggestion) latestSuggestion = val.suggestion;
        if (val.delivered_instructions && val.message) latestInstruction = val.message;
      }
    });
  });

  // Summarize recommended next step or suggestion for quick scanning
  if (summary) {
    if (latestSuggestion) {
      const rec = el('div', 'summary-line');
      rec.textContent = `Recommendation: ${latestSuggestion}`;
      summary.appendChild(rec);
    } else if (latestInstruction) {
      const rec = el('div', 'summary-line');
      rec.textContent = `Instruction: ${latestInstruction}`;
      summary.appendChild(rec);
    } else if (lastReflection && lastReflection.repair_action) {
      const ra = lastReflection.repair_action;
      const rec = el('div', 'summary-line');
      rec.textContent = `Repair Action: ${ra.tool_name}(${JSON.stringify(ra.arguments || {})})`;
      summary.appendChild(rec);
    } else if (lastAction) {
      const rec = el('div', 'summary-line');
      rec.textContent = `Next Action: ${lastAction.tool}(${JSON.stringify(lastAction.arguments || lastAction.args || {})})`;
      summary.appendChild(rec);
    }
    if (lastReflection && lastReflection.why) {
      const why = el('div', 'summary-line');
      why.textContent = `Reflection: ${lastReflection.why}`;
      summary.appendChild(why);
    }
  }
}

function renderJSON(state) {
  const out = document.getElementById('jsonOut');
  out.textContent = JSON.stringify(state, null, 2);
}

async function loadScenarios() {
  try {
    const sel = document.getElementById('scenarioSel');
    sel.innerHTML = '';
    const list = await api('/scenarios');
    list.forEach((s, i) => {
      const opt = el('option');
      opt.value = s.prompt;
      opt.textContent = `${s.name}`;
      if (i === 0) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.warn('No scenarios available:', e.message);
  }
}

async function warmModel() {
  const body = { keep_alive: '30m' };
  try {
    const res = await api('/warmup', { method: 'POST', body });
    alert('Warmup OK');
    console.log('warmup', res);
  } catch (e) {
    alert('Warmup failed: ' + e.message);
  }
}

async function solve() {
  const disruption = document.getElementById('prompt').value.trim();
  const offline = document.getElementById('offlineToggle').checked;
  if (!disruption) {
    alert('Please enter a description.');
    return;
  }
  const body = { disruption, offline };
  try {
    const state = await api('/solve', { method: 'POST', body });
    renderTrace(state);
    renderJSON(state);
  } catch (e) {
    renderJSON({ error: e.message });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadScenarios();
  document.getElementById('loadScenario').addEventListener('click', () => {
    const sel = document.getElementById('scenarioSel');
    document.getElementById('prompt').value = sel.value || '';
  });
  document.getElementById('warmBtn').addEventListener('click', warmModel);
  document.getElementById('solveBtn').addEventListener('click', solve);
});

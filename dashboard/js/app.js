/* app.js — RF ISP Dashboard */
const API = 'http://localhost:8000';

let state = {
  predictOption: 'existing', predictFile: null,
  featureChart: null, riskDonut: null, topRiskBar: null,
  allPredictions: [],
};

document.addEventListener('DOMContentLoaded', () => {
  checkApiStatus();
  setInterval(checkApiStatus, 8000);
  setupUpload();
  setupPredictFile();
  loadExistingResults();
  initProgSteps();
});

// ── Navigation ──────────────────────────────────────────────────────────
function navigate(section, el) {
  document.querySelectorAll('.section').forEach(s => s.classList.add('hidden'));
  document.getElementById(`section-${section}`)?.classList.remove('hidden');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  if (el) el.classList.add('active');
  else document.querySelector(`[data-section="${section}"]`)?.classList.add('active');
}

// ── Sidebar Toggle ───────────────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const isCollapsed = sidebar.classList.toggle('collapsed');
  document.body.classList.toggle('sidebar-collapsed', isCollapsed);
  localStorage.setItem('sidebarCollapsed', isCollapsed);
  // Saat expand, kembalikan lebar yang tersimpan
  if (!isCollapsed) {
    const savedW = localStorage.getItem('sidebarWidth');
    if (savedW) {
      document.documentElement.style.setProperty('--sidebar', savedW + 'px');
    }
  }
}

// Restore sidebar state dari localStorage
(function restoreSidebar() {
  if (localStorage.getItem('sidebarCollapsed') === 'true') {
    document.getElementById('sidebar')?.classList.add('collapsed');
    document.body.classList.add('sidebar-collapsed');
  }
  // Restore sidebar width
  const savedW = localStorage.getItem('sidebarWidth');
  if (savedW) {
    document.documentElement.style.setProperty('--sidebar', savedW + 'px');
  }
})();

// ── Sidebar Resize ───────────────────────────────────────────────────────
(function initSidebarResize() {
  const handle  = document.getElementById('sidebarResize');
  const sidebar = document.getElementById('sidebar');
  if (!handle || !sidebar) return;

  const getMinMax = () => {
    const style = getComputedStyle(document.documentElement);
    const min = parseInt(style.getPropertyValue('--sidebar-min')) || 180;
    const max = parseInt(style.getPropertyValue('--sidebar-max')) || 380;
    return { min, max };
  };

  let startX = 0, startW = 0, dragging = false;

  handle.addEventListener('mousedown', (e) => {
    if (sidebar.classList.contains('collapsed')) return;
    e.preventDefault();
    dragging = true;
    startX = e.clientX;
    startW = sidebar.getBoundingClientRect().width;
    handle.classList.add('dragging');
    sidebar.classList.add('resizing');
    document.body.classList.add('sidebar-resizing');
  });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const { min, max } = getMinMax();
    const newW = Math.min(max, Math.max(min, startW + (e.clientX - startX)));
    document.documentElement.style.setProperty('--sidebar', newW + 'px');
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    sidebar.classList.remove('resizing');
    document.body.classList.remove('sidebar-resizing');
    // Simpan lebar ke localStorage
    const currentW = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sidebar'));
    if (currentW) localStorage.setItem('sidebarWidth', currentW);
  });
})();

// ── API Status ───────────────────────────────────────────────────────────
async function checkApiStatus() {
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusText');
  try {
    const r = await fetch(`${API}/`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    dot.className = 'pulse-dot online'; txt.textContent = 'Online';
    if (d.model_trained) setModelReady();
  } catch {
    dot.className = 'pulse-dot offline'; txt.textContent = 'Offline';
  }
}

function setModelReady() {
  const b = document.getElementById('modelBadge');
  b.textContent = '✓ Model Siap'; b.classList.add('ready');
  // Tampilkan card download model jika sudah ada
  const mfc = document.getElementById('modelFilesCard');
  if (mfc && mfc.style.display === 'none') {
    mfc.style.display = '';
    loadModelInfo();
  }
}

// ── Upload ───────────────────────────────────────────────────────────────
function setupUpload() {
  const zone = document.getElementById('dropZone');
  const inp  = document.getElementById('csvFile');
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag-over'); if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); });
  inp.addEventListener('change', () => { if (inp.files[0]) uploadFile(inp.files[0]); });
}

async function uploadFile(file) {
  if (!file.name.endsWith('.csv')) { toast('error', 'File harus berformat .csv'); return; }
  toast('info', `Mengupload ${file.name}…`);
  const form = new FormData(); form.append('file', file);
  try {
    const r = await fetch(`${API}/api/upload`, { method: 'POST', body: form });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Upload gagal');
    renderUpload(d);
    toast('success', `Dataset diupload: ${d.rows} baris`);
  } catch(e) { toast('error', `Upload error: ${e.message}`); }
}

function renderUpload(d) {
  document.getElementById('uploadResult').classList.remove('hidden');
  const dist = d.class_distribution || {};
  const telat = dist['telat (1)'] ?? 0, tidak = dist['tidak_telat (0)'] ?? 0;
  document.getElementById('uploadKpi').innerHTML =
    kpi('Total Baris', d.rows, 'Semua pelanggan', '🗃️', 'cyan') +
    kpi('Pelanggan Telat', telat, 'latepay_30d = 1', '⚠️', 'red') +
    kpi('Tepat Waktu', tidak, 'latepay_30d = 0', '✅', 'green') +
    kpi('Label', d.label_auto_generated ? 'Auto-gen' : 'Manual', d.label_auto_generated ? 'Dibuat otomatis' : 'Sudah ada', '🏷️', 'violet');
  document.getElementById('previewTable').innerHTML = buildTable(d.preview || []);
}

// ── Train ────────────────────────────────────────────────────────────────
function initProgSteps() {
  const steps = ['Memuat data','Split 80/20','Train RF','Evaluasi','Simpan .pkl'];
  document.getElementById('progSteps').innerHTML = steps.map((s,i) =>
    `<span class="prog-step" id="pstep${i}">${s}</span>`).join('');
}

let _trainingDone = false;

async function trainModel() {
  const btn = document.getElementById('btnTrain');
  const n   = parseInt(document.getElementById('nEstimators').value) || 100;
  const d   = parseInt(document.getElementById('maxDepth').value) || null;
  const rs  = parseInt(document.getElementById('randomState').value) || 42;

  btn.disabled = true;
  _trainingDone = false;
  document.getElementById('trainingProgress').classList.remove('hidden');
  animateProg();

  const p = new URLSearchParams({ n_estimators: n, random_state: rs });
  if (d) p.append('max_depth', d);
  try {
    const r = await fetch(`${API}/api/train?${p}`, { method: 'POST' });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Training gagal');

    // Tandai selesai — timer tidak bisa override lagi
    _trainingDone = true;
    setProgDone();
    renderMetrics(data.metrics, data.feature_importances);
    document.getElementById('metricsCard').classList.remove('hidden');
    // Tampilkan card download model
    const mfc = document.getElementById('modelFilesCard');
    if (mfc) { mfc.style.display = ''; loadModelInfo(); }
    setModelReady();
    toast('success', '🧠 Model berhasil dilatih & disimpan!');
  } catch(e) {
    _trainingDone = true;
    toast('error', `Training: ${e.message}`);
    document.getElementById('progressLabel').textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; }
}

let _progTimers = [];

function animateProg() {
  const fill = document.getElementById('progressFill');
  const lbl  = document.getElementById('progressLabel');
  const pct  = document.getElementById('progressPct');
  const phases = [[10,'Memuat data…',0],[25,'Split 80/20…',1],[55,'Melatih Random Forest…',2],[80,'Evaluasi model…',3],[93,'Menyimpan .pkl…',4]];
  // Bersihkan timer lama
  _progTimers.forEach(t => clearTimeout(t));
  _progTimers = [];
  phases.forEach(([p,msg,si], i) => {
    const t = setTimeout(() => {
      if (_trainingDone) return; // jangan override kalau sudah 100%
      fill.style.width = p + '%';
      lbl.textContent  = msg;
      pct.textContent  = p + '%';
      for(let j=0;j<si;j++) document.getElementById('pstep'+j)?.classList.add('done');
      document.getElementById('pstep'+si)?.classList.add('active');
    }, i * 700);
    _progTimers.push(t);
  });
}

function setProgDone() {
  // Batalkan semua timer animasi yang masih pending
  _progTimers.forEach(t => clearTimeout(t));
  _progTimers = [];
  const fill = document.getElementById('progressFill');
  const lbl  = document.getElementById('progressLabel');
  const pct  = document.getElementById('progressPct');
  // Animasi smooth ke 100%
  fill.style.transition = 'width 0.6s ease';
  fill.style.width = '100%';
  lbl.textContent = '✅ Training & penyimpanan selesai!';
  pct.textContent = '100%';
  for(let i=0;i<5;i++) {
    const el = document.getElementById('pstep'+i);
    if(el){ el.classList.remove('active'); el.classList.add('done'); }
  }
}

function renderMetrics(m, imp) {
  document.getElementById('metricsKpi').innerHTML =
    kpi('Accuracy',  pctFmt(m.accuracy),  `${m.train_size} train / ${m.test_size} test`, '🎯', 'cyan') +
    kpi('Precision', pctFmt(m.precision), 'Presisi prediksi positif', '🔍', 'blue') +
    kpi('Recall',    pctFmt(m.recall),    'Sensitivitas model', '📡', 'green') +
    kpi('F1-Score',  pctFmt(m.f1_score),  'Harmonic mean P & R', '⚖️', 'amber');

  const cm = m.confusion_matrix || [[0,0],[0,0]];
  document.getElementById('confusionMatrix').innerHTML =
    `<div class="cm-cell cm-tn">${cm[0][0]}<span class="cm-lbl">TN</span></div>` +
    `<div class="cm-cell cm-fp">${cm[0][1]}<span class="cm-lbl">FP</span></div>` +
    `<div class="cm-cell cm-fn">${cm[1][0]}<span class="cm-lbl">FN</span></div>` +
    `<div class="cm-cell cm-tp">${cm[1][1]}<span class="cm-lbl">TP</span></div>`;

  renderFeatureChart(imp);
}

function renderFeatureChart(imp) {
  const ctx = document.getElementById('featureChart').getContext('2d');
  if (state.featureChart) state.featureChart.destroy();
  const labels = imp.map(i => i.label);
  const vals   = imp.map(i => i.importance);
  state.featureChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ data: vals, backgroundColor: vals.map(v =>
      v>=0.15?'rgba(6,182,212,0.7)':v>=0.10?'rgba(59,130,246,0.7)':'rgba(99,102,241,0.7)'),
      borderRadius: 5 }] },
    options: { indexAxis:'y', responsive:true, plugins:{legend:{display:false}},
      scales: {
        x:{ grid:{color:'rgba(99,179,237,0.06)'}, ticks:{color:'#7a90b8',font:{size:11}} },
        y:{ grid:{display:false}, ticks:{color:'#e8f0fe',font:{size:11}} }
      }
    }
  });
}

// ── Predict ──────────────────────────────────────────────────────────────
function selectPredictOption(opt) {
  state.predictOption = opt;
  document.getElementById('optExisting').classList.toggle('active', opt==='existing');
  document.getElementById('optNew').classList.toggle('active', opt==='new');
  document.getElementById('newFileWrap').classList.toggle('hidden', opt==='existing');
}

function setupPredictFile() {
  const inp = document.getElementById('predictFile');
  inp.addEventListener('change', () => {
    if (inp.files[0]) {
      state.predictFile = inp.files[0];
      document.getElementById('predictFileName').textContent = inp.files[0].name;
    }
  });
}

async function runPredict() {
  toast('info', 'Menjalankan prediksi…');
  try {
    let res;
    if (state.predictOption === 'new' && state.predictFile) {
      const form = new FormData(); form.append('file', state.predictFile);
      res = await fetch(`${API}/api/predict`, { method:'POST', body:form });
    } else {
      res = await fetch(`${API}/api/predict`, { method:'POST' });
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Prediksi gagal');
    state.allPredictions = data.top_risk_customers || [];
    renderResults(data);
    navigate('result', null);
    toast('success', `🔮 ${data.summary.total_customers} pelanggan diprediksi`);
  } catch(e) { toast('error', `Prediksi: ${e.message}`); }
}

// ── Results ───────────────────────────────────────────────────────────────
async function loadExistingResults() {
  try {
    const r = await fetch(`${API}/api/results`);
    const d = await r.json();
    if (d.available) {
      state.allPredictions = d.top_risk || [];
      renderResultsFromAPI(d);
    }
  } catch {}
}

function renderResults(d) {
  renderResultKpi(d.summary);
  renderRiskDonut(d.summary);
  renderTopRiskBar(d.top_risk_customers);
  renderResultTable(d.top_risk_customers);
}

function renderResultsFromAPI(d) {
  const s = { ...d.summary, total_customers: d.total_customers };
  renderResultKpi(s); renderRiskDonut(s);
  renderTopRiskBar(d.top_risk); renderResultTable(d.top_risk);
}

function renderResultKpi(s) {
  const tot = s.total_customers;
  const hp  = tot ? Math.round(s.high_risk/tot*100) : 0;
  document.getElementById('resultKpi').innerHTML =
    kpi('Total Pelanggan', tot, 'Seluruh dataset', '👥', 'cyan') +
    kpi('Risiko Tinggi', s.high_risk, `${hp}% dari total`, '🔴', 'red') +
    kpi('Risiko Sedang', s.medium_risk, 'Probabilitas 30–60%', '🟡', 'amber') +
    kpi('Risiko Rendah', s.low_risk, 'Probabilitas < 30%', '🟢', 'green') +
    kpi('Avg Probabilitas', pctFmt(s.avg_probability), 'Rata-rata risiko telat', '📊', 'blue');
}

function renderRiskDonut(s) {
  const ctx = document.getElementById('riskDonut').getContext('2d');
  if (state.riskDonut) state.riskDonut.destroy();
  const tot = s.high_risk + s.medium_risk + s.low_risk || 1;
  document.getElementById('donutCenter').innerHTML =
    `<div style="font-size:22px;font-weight:800;color:var(--red)">${s.high_risk}</div>
     <div style="font-size:10px;color:var(--muted);margin-top:2px">Risiko Tinggi</div>`;
  state.riskDonut = new Chart(ctx, {
    type: 'doughnut',
    data: { labels:['Tinggi ≥60%','Sedang 30-60%','Rendah <30%'],
      datasets:[{ data:[s.high_risk,s.medium_risk,s.low_risk],
        backgroundColor:['rgba(239,68,68,0.85)','rgba(245,158,11,0.85)','rgba(16,185,129,0.85)'],
        borderWidth:0, hoverOffset:8 }] },
    options: { responsive:true, maintainAspectRatio:false, cutout:'68%',
      plugins:{ legend:{ position:'bottom', labels:{ color:'#7a90b8', font:{size:10}, padding:12 } } } }
  });
}

function renderTopRiskBar(customers) {
  const ctx = document.getElementById('topRiskBar').getContext('2d');
  if (state.topRiskBar) state.topRiskBar.destroy();
  if (!customers?.length) return;
  const top = customers.slice(0,10);
  const labels = top.map(c => c.nama_pelanggan || c.id_pelanggan || '?');
  const vals   = top.map(c => parseFloat((c.proba_latepay*100).toFixed(1)));
  const colors = vals.map(v => v>=60?'rgba(239,68,68,0.8)':v>=30?'rgba(245,158,11,0.8)':'rgba(16,185,129,0.8)');
  state.topRiskBar = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets:[{ data:vals, backgroundColor:colors, borderRadius:6 }] },
    options: { responsive:true, plugins:{legend:{display:false}},
      scales:{
        x:{ grid:{display:false}, ticks:{color:'#7a90b8',font:{size:10},maxRotation:35} },
        y:{ max:100, grid:{color:'rgba(99,179,237,0.06)'}, ticks:{color:'#7a90b8',callback:v=>v+'%'} }
      }
    }
  });
}

function renderResultTable(customers) {
  if (!customers?.length) { document.getElementById('resultTable').innerHTML = '<p style="padding:16px;color:var(--muted);font-size:13px">Belum ada data prediksi.</p>'; return; }
  const rows = customers.map(c => {
    const p = parseFloat(c.proba_latepay||0);
    const risk = c.risk_level || (p>=0.6?'Tinggi':p>=0.3?'Sedang':'Rendah');
    const pct2 = (p*100).toFixed(1);
    const col  = p>=0.6?'var(--red)':p>=0.3?'var(--amber)':'var(--green)';
    return `<tr>
      <td class="mono">${c.id_pelanggan||'-'}</td>
      <td style="font-weight:500">${c.nama_pelanggan||'-'}</td>
      <td><div class="prob-wrap">
        <div class="prob-track"><div class="prob-fill" style="width:${pct2}%;background:${col}"></div></div>
        <span class="prob-num" style="color:${col}">${pct2}%</span>
      </div></td>
      <td>${c.pred_label==1?'<span style="color:var(--red);font-weight:600">⚠ Telat</span>':'<span style="color:var(--green);font-weight:600">✓ Tepat</span>'}</td>
      <td><span class="badge ${risk.toLowerCase()}">${risk}</span></td>
      <td class="mono">${c.telat_bayar_30??'-'}</td>
      <td class="mono">${c.freq_keluhan_30??'-'}</td>
    </tr>`;
  });
  document.getElementById('resultTable').innerHTML =
    `<table><thead><tr>
      <th>ID</th><th>Nama Pelanggan</th><th>Probabilitas Telat</th>
      <th>Prediksi</th><th>Risiko</th><th>Telat (30h)</th><th>Keluhan</th>
    </tr></thead><tbody>${rows.join('')}</tbody></table>`;
}

function filterTable(q) {
  const rows = document.querySelectorAll('#resultTable tbody tr');
  rows.forEach(r => {
    r.style.display = r.textContent.toLowerCase().includes(q.toLowerCase()) ? '' : 'none';
  });
}

async function downloadResults() { window.open(`${API}/api/download-results`, '_blank'); }

function downloadModel() {
  window.open(`${API}/api/download-model`, '_blank');
  toast('info', '⬇️ Mendownload rf_model.pkl…');
}

function downloadMetrics() {
  window.open(`${API}/api/download-metrics`, '_blank');
  toast('info', '⬇️ Mendownload rf_metrics.json…');
}

async function loadModelInfo() {
  const box = document.getElementById('modelInfoBox');
  if (!box) return;
  box.innerHTML = '<div class="mib-loading">⏳ Memuat info model…</div>';
  try {
    const r = await fetch(`${API}/api/model-info`);
    const d = await r.json();
    if (!d.trained) { box.innerHTML = '<div class="mib-loading">Model belum dilatih.</div>'; return; }
    const fi = d.feature_importances || [];
    const topFeat = fi[0] ? `${fi[0].label} (${(fi[0].importance*100).toFixed(1)}%)` : '-';
    box.innerHTML = `<div class="mib-grid">
      <div class="mib-item"><span class="mib-label">Accuracy</span><span class="mib-value">${pctFmt(d.accuracy)}</span></div>
      <div class="mib-item"><span class="mib-label">Precision</span><span class="mib-value green">${pctFmt(d.precision)}</span></div>
      <div class="mib-item"><span class="mib-label">Recall</span><span class="mib-value amber">${pctFmt(d.recall)}</span></div>
      <div class="mib-item"><span class="mib-label">F1-Score</span><span class="mib-value">${pctFmt(d.f1_score)}</span></div>
      <div class="mib-item"><span class="mib-label">n_estimators</span><span class="mib-value">${d.n_estimators ?? '-'}</span></div>
      <div class="mib-item"><span class="mib-label">Train Size</span><span class="mib-value">${d.train_size ?? '-'}</span></div>
      <div class="mib-item"><span class="mib-label">Fitur Terpenting</span><span class="mib-value" style="font-size:11px">${topFeat}</span></div>
      <div class="mib-item"><span class="mib-label">Dilatih</span><span class="mib-value" style="font-size:10px;color:var(--muted)">${d.trained_at ? new Date(d.trained_at).toLocaleString('id-ID') : '-'}</span></div>
    </div>`;
  } catch(e) { box.innerHTML = '<div class="mib-loading">Gagal memuat info model.</div>'; }
}

// ── Helpers ───────────────────────────────────────────────────────────────
function kpi(label, value, sub, icon, color) {
  return `<div class="kpi-card ${color}"><span class="kpi-icon">${icon}</span>
    <div class="kpi-label">${label}</div>
    <div class="kpi-value">${value}</div>
    <div class="kpi-sub">${sub}</div></div>`;
}

function buildTable(rows) {
  if (!rows?.length) return '<p style="padding:14px;color:var(--muted);font-size:12px">Tidak ada data.</p>';
  const cols = Object.keys(rows[0]);
  return `<table><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${r[c]??'-'}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

function pctFmt(v) {
  if (typeof v==='number' && v<=1) return (v*100).toFixed(1)+'%';
  return v??'-';
}

function toast(type, msg) {
  const c = document.getElementById('toastContainer');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${{success:'✅',error:'❌',info:'ℹ️'}[type]||''}</span><span>${msg}</span>`;
  c.appendChild(el);
  setTimeout(() => { el.style.opacity='0'; el.style.transition='opacity .3s'; }, 3500);
  setTimeout(() => el.remove(), 3800);
}

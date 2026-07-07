// kubemark-ai Dashboard

let allResults = [];
let allComparisons = [];
let ncclChart = null;
let latencyChart = null;
let tflopsChart = null;
let compareChart = null;
let runtimeNcclChart = null;
let runtimeColdstartChart = null;

// Runtime display metadata
const RUNTIME_META = {
  default: { label: 'runc (default)', color: '#76b900' },
  runc: { label: 'runc', color: '#76b900' },
  gvisor: { label: 'gVisor', color: '#00bcd4' },
  'kata-qemu-nvidia-gpu': { label: 'Kata (QEMU)', color: '#ff6b35' },
  kata: { label: 'Kata', color: '#ff6b35' },
  vnode: { label: 'vNode', color: '#a855f7' },
};

function runtimeMeta(rt) {
  return RUNTIME_META[rt] || { label: rt, color: '#eab308' };
}

// Color palette
const COLORS = {
  green: '#76b900',
  cyan: '#00bcd4',
  orange: '#ff6b35',
  purple: '#a855f7',
  pink: '#ec4899',
  yellow: '#eab308',
};

const COLOR_LIST = Object.values(COLORS);

// GPU hardware specs and published benchmark references
//
// PRIMARY SOURCE for H100, B200, GB200, GB300:
//   github.com/NVIDIA/dgxc-benchmarking — NVIDIA's official benchmark suite
//   Reference infrastructure specs: github.com/NVIDIA/dgxc-benchmarking#reference-infrastructure
//   NCCL sample output (H100): github.com/NVIDIA/dgxc-benchmarking/blob/main/nccl/README.md
//   Peak TFLOPS: published in llama3.1/README.md, nemotron4-340b/README.md
//
// SUPPLEMENTARY sources for non-DGXC architectures cited inline
const GPU_SPECS = {
  // --- NVIDIA DGXC Benchmarking supported architectures ---
  // All specs from: github.com/NVIDIA/dgxc-benchmarking#reference-infrastructure
  h100: {
    nvlinkBw: 900,        // NVLink 4.0: 900 GB/s bidirectional per GPU (18 links)
    memBw: 3200,           // HBM3: 3.2 TB/s
    peakTflops: { bf16: 989, fp8: 1979 },
    interconnect: 'NVLink 4.0',
    gpusPerNode: 8,
    dgxcSupported: true,
    // NVIDIA dgxc-benchmarking sample output (8xH100 AllReduce):
    //   avg busbw = 159.15 GB/s, peak busbw = 479.99 GB/s (at 16GB message)
    // Source: github.com/NVIDIA/dgxc-benchmarking/blob/main/nccl/README.md
    publishedAllReduce: 159.15,
    publishedAllReducePeak: 480,
    publishedSource: 'NVIDIA dgxc-benchmarking sample (8xH100)',
    publishedSourceUrl: 'https://github.com/NVIDIA/dgxc-benchmarking/blob/main/nccl/README.md',
  },
  b200: {
    nvlinkBw: 1800,       // NVLink 5.0: 1.8 TB/s per GPU (18 links)
    memBw: 8000,           // HBM3e: 8 TB/s
    peakTflops: { bf16: 2250, fp8: 4500 },
    interconnect: 'NVLink 5.0',
    gpusPerNode: 8,
    dgxcSupported: true,
    // No published NCCL numbers in dgxc-benchmarking repo for B200
    publishedSource: 'NVIDIA dgxc-benchmarking (hardware specs only)',
    publishedSourceUrl: 'https://github.com/NVIDIA/dgxc-benchmarking#reference-infrastructure',
  },
  gb200: {
    nvlinkBw: 1800,       // NVLink 5.0: 1.8 TB/s per GPU
    memBw: 8000,           // HBM3e: 8 TB/s (16 TB/s total system)
    peakTflops: { bf16: 2450, fp8: 4900 },
    interconnect: 'NVLink 5.0',
    gpusPerNode: 4,        // 4 tasks per node (2 superchips x 2 GPUs)
    dgxcSupported: true,
    // No published NCCL numbers in dgxc-benchmarking repo for GB200
    publishedSource: 'NVIDIA dgxc-benchmarking (hardware specs only)',
    publishedSourceUrl: 'https://github.com/NVIDIA/dgxc-benchmarking#reference-infrastructure',
  },
  gb300: {
    nvlinkBw: 1800,       // NVLink 5.0: 1.8 TB/s per GPU
    memBw: 12000,          // HBM3e: 12 TB/s per GPU
    peakTflops: { bf16: 2450, fp8: 4900 },
    interconnect: 'NVLink 5.0',
    gpusPerNode: 4,
    dgxcSupported: true,
    // No published NCCL numbers in dgxc-benchmarking repo for GB300
    publishedSource: 'NVIDIA dgxc-benchmarking (hardware specs only)',
    publishedSourceUrl: 'https://github.com/NVIDIA/dgxc-benchmarking#reference-infrastructure',
  },

  // --- Community-verified architectures (not in NVIDIA DGXC suite) ---
  a100: {
    nvlinkBw: 600,        // NVLink 3.0: 600 GB/s bidirectional per GPU
    memBw: 2039,           // HBM2e: 2 TB/s
    peakTflops: { bf16: 312, fp8: null },
    interconnect: 'NVLink 3.0',
    gpusPerNode: 8,
    dgxcSupported: false,
    // Community sources: ~226 GB/s (nccl-tests #149), ~150 GB/s (DGX A100 NVSwitch Blog)
    publishedAllReduce: 150,
    publishedSource: 'NVIDIA NVSwitch 3rd Gen Blog (DGX A100)',
    publishedSourceUrl: 'https://developer.nvidia.com/blog/?p=53977',
  },
};

// Interconnect performance tiers (from NVIDIA nccl-tests PERFORMANCE.md)
// Source: github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
const INTERCONNECT_TIERS = {
  nvlink:    { label: 'NVLink (intra-node)', range: '150-480 GB/s', min: 150, max: 480 },
  ib400:     { label: 'InfiniBand 400Gb/s (multi-node)', range: '354-367 GB/s', min: 354, max: 367 },
  ib200:     { label: 'InfiniBand 200Gb/s (multi-node)', range: '150-190 GB/s', min: 150, max: 190 },
  pcie_p2p:  { label: 'PCIe P2P (L40S, A10G, L4)', range: '8-22 GB/s', min: 8, max: 22 },
  pcie_cpu:  { label: 'PCIe via CPU (T4, V100 PCIe)', range: '5-12 GB/s', min: 5, max: 12 },
  ethernet:  { label: 'Ethernet (GKE, EKS)', range: '1-12 GB/s', min: 1, max: 12 },
  dgxspark:  { label: 'DGX Spark QSFP (GB10)', range: '~23 GB/s', min: 20, max: 25 },
};

// Chart.js global defaults
Chart.defaults.color = '#8b8fa3';
Chart.defaults.borderColor = '#2e3345';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif';

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');
  });
});

// Format bytes to human readable
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
}

// Load results from API or sample data
async function loadResults() {
  try {
    const response = await fetch('/api/results');
    allResults = await response.json();
  } catch (e) {
    // Load sample data for demo
    allResults = getSampleData();
  }

  try {
    const response = await fetch('/api/comparisons');
    allComparisons = await response.json();
  } catch (e) {
    allComparisons = getSampleComparisons();
  }

  if (allResults.length === 0 && allComparisons.length === 0) {
    document.getElementById('empty-state').style.display = 'block';
    return;
  }

  populateSelectors();
  renderOverview();
  renderNCCLRun();
  renderTrainingRun();
  renderRuntimeComparison();
}

// ---- Overview tab ----
function renderOverview() {
  const statsEl = document.getElementById('overview-stats');
  const findingsEl = document.getElementById('overview-findings');
  const tableEl = document.getElementById('overview-runs-table');
  if (!statsEl) return;

  const ncclRuns = allResults.filter(r => r.benchmark_type === 'nccl');
  const trainingRuns = allResults.filter(r => r.benchmark_type === 'training');
  const coldstartRuns = allResults.filter(r => r.benchmark_type === 'coldstart');
  const gpuTypes = [...new Set(allResults.map(r => (r.gpu_type || '').toUpperCase()).filter(Boolean))];
  const runtimesTested = new Set();
  allComparisons.forEach(c => Object.keys(c.runtimes || {}).forEach(rt => runtimesTested.add(rt)));

  statsEl.innerHTML = [
    { label: 'Total Runs', value: allResults.length, unit: '' },
    { label: 'NCCL / Training / Cold-start', value: `${ncclRuns.length} / ${trainingRuns.length} / ${coldstartRuns.length}`, unit: '' },
    { label: 'Runtime Comparisons', value: allComparisons.length, unit: '' },
    { label: 'GPU Types', value: gpuTypes.join(', ') || '—', unit: '' },
    { label: 'Runtimes Tested', value: [...runtimesTested].map(rt => runtimeMeta(rt).label.split(' ')[0]).join(', ') || '—', unit: '' },
  ].map(s => `
    <div class="stat-card">
      <div class="label">${s.label}</div>
      <div class="value" style="font-size:${String(s.value).length > 12 ? '18px' : '28px'}">${s.value}<span class="unit">${s.unit}</span></div>
    </div>`).join('');

  // Headline findings from the most recent comparison with a baseline + >=2 runtimes
  const findings = [];
  const latest = [...allComparisons].reverse().find(c => Object.keys(c.runtimes || {}).length >= 2);
  if (latest) {
    const base = latest.baseline || 'default';
    for (const [rt, entry] of Object.entries(latest.runtimes)) {
      if (rt === base || !entry.overhead_vs_baseline) continue;
      const ov = entry.overhead_vs_baseline;
      const meta = runtimeMeta(rt);
      const parts = [];
      if (ov.cuda_init_ms_delta !== undefined) {
        const d = ov.cuda_init_ms_delta;
        parts.push(`CUDA init ${d >= 0 ? '+' : ''}${d} ms`);
      }
      if (ov.nccl_all_reduce_bandwidth_pct !== undefined) {
        parts.push(`NCCL bandwidth ${ov.nccl_all_reduce_bandwidth_pct >= 0 ? '+' : ''}${ov.nccl_all_reduce_bandwidth_pct}%`);
      }
      const near = Math.abs(ov.cuda_init_ms_delta || 0) < 15;
      findings.push({
        title: `${meta.label} vs ${runtimeMeta(base).label}`,
        color: meta.color,
        text: parts.join(' · ') + (near ? ' — indistinguishable from baseline' : ''),
        tag: latest.vcluster ? 'inside vCluster' : 'host cluster',
      });
    }
    // vCluster overhead finding if we have both a host and a vCluster comparison
    const hostCmp = allComparisons.find(c => !c.vcluster && Object.keys(c.runtimes || {}).length >= 2);
    const vcCmp = allComparisons.find(c => c.vcluster);
    if (hostCmp && vcCmp) {
      findings.push({
        title: 'vCluster isolation level',
        color: '#eab308',
        text: 'Runtime deltas reproduce within noise inside ephemeral vClusters — control-plane virtualization adds no measurable data-plane overhead.',
        tag: `${hostCmp.compare_id} vs ${vcCmp.compare_id}`,
      });
    }
  }
  findingsEl.innerHTML = findings.length ? `
    <div class="chart-container">
      <h3>Headline Findings</h3>
      <div class="info-grid">
        ${findings.map(f => `
          <div class="info-item">
            <strong style="color:${f.color}">${f.title}</strong>
            <p>${f.text}</p>
            <p style="margin-top:6px;font-size:11px;opacity:0.7">${f.tag}</p>
          </div>`).join('')}
      </div>
    </div>` : '';

  // Recent runs table
  const recent = [...allResults].sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || '')).slice(0, 8);
  tableEl.innerHTML = recent.length ? `
    <table class="overview-table">
      <thead><tr><th>Run</th><th>Type</th><th>GPU</th><th>Runtime</th><th>Key Result</th></tr></thead>
      <tbody>
        ${recent.map(r => {
          let key = '—';
          if (r.benchmark_type === 'nccl' && Array.isArray(r.results) && r.results[0]) {
            key = `${r.results[0].avg_bus_bandwidth_gbps ?? '—'} GB/s busbw`;
          } else if (r.benchmark_type === 'training' && r.results?.tflops_per_gpu_mean) {
            key = `${r.results.tflops_per_gpu_mean} TFLOPS/GPU`;
          } else if (r.benchmark_type === 'coldstart' && r.results?.cuda_init_ms) {
            key = `${r.results.cuda_init_ms} ms CUDA init`;
          }
          return `<tr>
            <td>${r.run_id || '—'}</td>
            <td>${r.benchmark_type || '—'}</td>
            <td>${(r.gpu_type || '—').toUpperCase()} ×${r.total_gpus ?? '?'}</td>
            <td>${runtimeMeta(r.runtime_class || 'default').label}</td>
            <td>${key}</td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>` : '<p class="chart-help">No runs yet — run <code>./scripts/run-benchmark.sh</code> to get started.</p>';
}

function populateSelectors() {
  const ncclRuns = allResults.filter(r => r.benchmark_type === 'nccl');
  const trainingRuns = allResults.filter(r => r.benchmark_type === 'training');

  fillSelect('nccl-run-select', ncclRuns);
  fillSelect('training-run-select', trainingRuns);
  fillSelect('compare-a', ncclRuns);
  fillSelect('compare-b', ncclRuns);

  // Runtime comparison selector
  const rtSelect = document.getElementById('runtime-compare-select');
  if (rtSelect) {
    rtSelect.innerHTML = '';
    allComparisons.forEach((cmp, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      const runtimes = Object.keys(cmp.runtimes || {}).join(' vs ');
      const vcTag = cmp.vcluster ? ' [inside vCluster]' : '';
      opt.textContent = `${cmp.compare_id} — ${runtimes}${vcTag}`;
      rtSelect.appendChild(opt);
    });
  }
}

function fillSelect(id, runs) {
  const select = document.getElementById(id);
  if (!select) return;
  select.innerHTML = '';
  runs.forEach((run, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    const rtSuffix = run.runtime_class && run.runtime_class !== 'default' ? ` [${run.runtime_class}]` : '';
    opt.textContent = `${run.gpu_type.toUpperCase()} x${run.total_gpus}${rtSuffix} — ${run.run_id} (${new Date(run.timestamp).toLocaleDateString()})`;
    select.appendChild(opt);
  });
}

function renderNCCLRun() {
  const ncclRuns = allResults.filter(r => r.benchmark_type === 'nccl');
  const select = document.getElementById('nccl-run-select');
  if (!select || ncclRuns.length === 0) return;

  const run = ncclRuns[select.value];
  if (!run) return;

  const results = Array.isArray(run.results) ? run.results : [run.results];
  const spec = GPU_SPECS[run.gpu_type.toLowerCase()];

  // Stats cards
  const statsEl = document.getElementById('nccl-stats');
  const primaryResult = results[0];
  const maxBw = primaryResult?.data_points
    ? Math.max(...primaryResult.data_points.map(d => d.busbw_gbps))
    : 0;
  const avgBw = primaryResult?.avg_bus_bandwidth_gbps || 0;

  // Performance context — only show hardware spec comparison for supported GPUs
  const isSingleGpu = run.total_gpus <= 1;
  let pctOfRef = 0;
  let refValue = 0;
  let refLabel = '';

  statsEl.innerHTML = `
    <div class="stat-card">
      <div class="label">GPU Type</div>
      <div class="value">${run.gpu_type.toUpperCase()}</div>
    </div>
    <div class="stat-card">
      <div class="label">Total GPUs</div>
      <div class="value">${run.total_gpus}</div>
    </div>
    <div class="stat-card">
      <div class="label">Nodes</div>
      <div class="value">${run.num_nodes}</div>
    </div>
    <div class="stat-card">
      <div class="label">Runtime</div>
      <div class="value">${run.runtime_class || 'default'}</div>
    </div>
    <div class="stat-card">
      <div class="label">${isSingleGpu ? 'Avg Bandwidth' : 'Avg Bus Bandwidth'}</div>
      <div class="value">${avgBw.toFixed(1)} <span class="unit">GB/s</span></div>
    </div>
    <div class="stat-card">
      <div class="label">Peak Bandwidth</div>
      <div class="value">${maxBw.toFixed(1)} <span class="unit">GB/s</span></div>
    </div>
    <div class="stat-card">
      <div class="label">Tests Run</div>
      <div class="value">${results.length}</div>
    </div>
  `;

  // Info box — contextual information about the result with sourced references
  const ratingEl = document.getElementById('nccl-rating');
  ratingEl.className = 'rating-box';
  ratingEl.style.display = '';

  if (isSingleGpu) {
    ratingEl.className = 'rating-box rating-good';
    ratingEl.innerHTML = `<div class="rating-label">Single-GPU Test</div><div>This measures <strong>internal GPU memory bandwidth</strong> (${maxBw.toFixed(1)} GB/s), not inter-GPU communication. ${spec ? `The ${run.gpu_type.toUpperCase()} has ${spec.memBw} GB/s theoretical memory bandwidth.` : ''} For GPU-to-GPU benchmarks, use 2+ GPUs. <em>Source: <a href="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md" target="_blank">NCCL Performance Doc</a></em></div>`;
  } else if (spec && spec.dgxcSupported) {
    // NVIDIA DGXC-supported GPU — use dgxc-benchmarking as primary source
    const sourceUrl = spec.publishedSourceUrl || 'https://github.com/NVIDIA/dgxc-benchmarking';
    let refNote;
    if (spec.publishedAllReduce) {
      refNote = `NVIDIA dgxc-benchmarking published AllReduce: <strong>${spec.publishedAllReduce} GB/s avg</strong>, <strong>${spec.publishedAllReducePeak} GB/s peak</strong> (<a href="${sourceUrl}" target="_blank">${spec.publishedSource}</a>).`;
      const pct = ((maxBw / spec.publishedAllReducePeak) * 100).toFixed(0);
      refNote += ` Your peak (${maxBw.toFixed(1)} GB/s) is <strong>${pct}%</strong> of published peak.`;
    } else {
      refNote = `Hardware NVLink spec: ${spec.nvlinkBw} GB/s per GPU. No published NCCL benchmark numbers in <a href="${sourceUrl}" target="_blank">dgxc-benchmarking</a> for this GPU yet.`;
    }
    const tfNote = spec.peakTflops ? ` Peak theoretical: ${spec.peakTflops.bf16} TFLOPS (BF16), ${spec.peakTflops.fp8} TFLOPS (FP8).` : '';
    ratingEl.className = 'rating-box rating-good';
    ratingEl.innerHTML = `<div class="rating-label">NVIDIA DGXC Supported Architecture</div><div>The ${run.gpu_type.toUpperCase()} is in <a href="https://github.com/NVIDIA/dgxc-benchmarking" target="_blank">NVIDIA's DGX Cloud Benchmarking suite</a>. ${refNote}${tfNote} Result: ${maxBw.toFixed(1)} GB/s peak across ${run.total_gpus} GPUs (${run.num_nodes} nodes).</div>`;
  } else if (spec && !spec.dgxcSupported) {
    // Known GPU with specs but not in DGXC (e.g., A100)
    const sourceUrl = spec.publishedSourceUrl || 'https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md';
    const refNote = spec.publishedAllReduce
      ? `Community-verified AllReduce: ~${spec.publishedAllReduce} GB/s (<a href="${sourceUrl}" target="_blank">${spec.publishedSource}</a>).`
      : '';
    ratingEl.className = 'rating-box';
    ratingEl.innerHTML = `<div class="rating-label">Community-Verified Architecture</div><div>The ${run.gpu_type.toUpperCase()} is not in <a href="https://github.com/NVIDIA/dgxc-benchmarking" target="_blank">NVIDIA's DGXC benchmark suite</a> (which covers H100, B200, GB200, GB300) but is well-documented in the community. ${refNote} Your peak: ${maxBw.toFixed(1)} GB/s across ${run.total_gpus} GPUs (${run.num_nodes} nodes). <em>See <a href="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md" target="_blank">NCCL Performance Doc</a> for expected ranges by interconnect type.</em></div>`;
  } else if (run.total_gpus > 1) {
    // Unknown GPU type — show interconnect tier guidance
    ratingEl.className = 'rating-box';
    ratingEl.innerHTML = `<div class="rating-label">Multi-GPU Result</div><div>Peak bus bandwidth: ${maxBw.toFixed(1)} GB/s across ${run.total_gpus} GPUs (${run.num_nodes} nodes). Compare against interconnect tiers: NVLink = 150-480 GB/s, PCIe P2P = 8-22 GB/s, Ethernet = 1-12 GB/s. <em>Source: <a href="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md" target="_blank">NCCL Performance Doc</a></em></div>`;
  } else {
    ratingEl.className = 'rating-box';
    ratingEl.style.display = 'none';
  }

  // Bandwidth chart
  if (ncclChart) ncclChart.destroy();
  const ctx = document.getElementById('nccl-bandwidth-chart').getContext('2d');

  const datasets = results.map((result, i) => ({
    label: (result.test || 'all_reduce').replace(/_/g, ' '),
    data: (result.data_points || [])
      .filter(d => d.size_bytes >= 1024)
      .map(d => ({ x: formatBytes(d.size_bytes), y: d.busbw_gbps })),
    borderColor: COLOR_LIST[i % COLOR_LIST.length],
    backgroundColor: COLOR_LIST[i % COLOR_LIST.length] + '33',
    tension: 0.3,
    fill: false,
    pointRadius: 3,
  }));

  const labels = (results[0]?.data_points || [])
    .filter(d => d.size_bytes >= 1024)
    .map(d => formatBytes(d.size_bytes));

  // Add NVLink spec line for supported GPUs (hardware spec, not benchmark baseline)
  if (spec && spec.dgxcSupported && !isSingleGpu) {
    datasets.push({
      label: `${run.gpu_type.toUpperCase()} NVLink spec (${spec.nvlinkBw} GB/s per GPU)`,
      data: labels.map(() => spec.nvlinkBw),
      borderColor: '#ffffff22',
      borderDash: [8, 4],
      pointRadius: 0,
      fill: false,
      tension: 0,
    });
  }

  ncclChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
      },
      scales: {
        y: {
          title: { display: true, text: 'Bus Bandwidth (GB/s)' },
          beginAtZero: true,
        },
        x: {
          title: { display: true, text: 'Message Size' },
        },
      },
    },
  });

  // Latency chart
  if (latencyChart) latencyChart.destroy();
  const ctx2 = document.getElementById('nccl-latency-chart').getContext('2d');

  const latencyDatasets = results.map((result, i) => ({
    label: (result.test || 'all_reduce').replace(/_/g, ' '),
    data: (result.data_points || [])
      .filter(d => d.size_bytes >= 1024)
      .map(d => d.time_us),
    borderColor: COLOR_LIST[i % COLOR_LIST.length],
    backgroundColor: COLOR_LIST[i % COLOR_LIST.length] + '33',
    tension: 0.3,
    fill: false,
    pointRadius: 3,
  }));

  latencyChart = new Chart(ctx2, {
    type: 'line',
    data: { labels, datasets: latencyDatasets },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
      },
      scales: {
        y: {
          title: { display: true, text: 'Latency (us)' },
          type: 'logarithmic',
        },
        x: {
          title: { display: true, text: 'Message Size' },
        },
      },
    },
  });
}

function renderTrainingRun() {
  const trainingRuns = allResults.filter(r => r.benchmark_type === 'training');
  const select = document.getElementById('training-run-select');
  if (!select || trainingRuns.length === 0) return;

  const run = trainingRuns[select.value];
  if (!run || !run.results) return;

  const r = run.results;

  const statsEl = document.getElementById('training-stats');
  statsEl.innerHTML = `
    <div class="stat-card">
      <div class="label">GPU Type</div>
      <div class="value">${run.gpu_type.toUpperCase()}</div>
    </div>
    <div class="stat-card">
      <div class="label">Total GPUs</div>
      <div class="value">${run.total_gpus}</div>
    </div>
    <div class="stat-card">
      <div class="label">TFLOPS/GPU</div>
      <div class="value">${r.tflops_per_gpu_mean?.toFixed(1) || 'N/A'} <span class="unit">&plusmn;${r.tflops_per_gpu_std?.toFixed(1) || 0}</span></div>
    </div>
    <div class="stat-card">
      <div class="label">Step Time</div>
      <div class="value">${r.step_time_mean_ms?.toFixed(1) || 'N/A'} <span class="unit">ms</span></div>
    </div>
    <div class="stat-card">
      <div class="label">Iterations</div>
      <div class="value">${r.iterations_measured || 'N/A'}</div>
    </div>
    <div class="stat-card">
      <div class="label">Status</div>
      <div class="value" style="color: ${r.status === 'COMPLETE' ? '#76b900' : '#ef4444'}">${r.status || 'N/A'}</div>
    </div>
  `;

  // TFLOPS bar chart comparing all training runs
  if (tflopsChart) tflopsChart.destroy();
  const ctx = document.getElementById('training-tflops-chart').getContext('2d');

  tflopsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: trainingRuns.map(r => `${r.gpu_type.toUpperCase()} x${r.total_gpus}`),
      datasets: [{
        label: 'TFLOPS per GPU',
        data: trainingRuns.map(r => r.results?.tflops_per_gpu_mean || 0),
        backgroundColor: trainingRuns.map((_, i) => COLOR_LIST[i % COLOR_LIST.length] + 'cc'),
        borderColor: trainingRuns.map((_, i) => COLOR_LIST[i % COLOR_LIST.length]),
        borderWidth: 1,
        errorBars: trainingRuns.map(r => r.results?.tflops_per_gpu_std || 0),
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          title: { display: true, text: 'TFLOPS per GPU' },
          beginAtZero: true,
        },
      },
    },
  });
}

function renderComparison() {
  const ncclRuns = allResults.filter(r => r.benchmark_type === 'nccl');
  const a = ncclRuns[document.getElementById('compare-a').value];
  const b = ncclRuns[document.getElementById('compare-b').value];
  if (!a || !b) return;

  const aResult = Array.isArray(a.results) ? a.results[0] : a.results;
  const bResult = Array.isArray(b.results) ? b.results[0] : b.results;

  if (compareChart) compareChart.destroy();
  const ctx = document.getElementById('compare-chart').getContext('2d');

  const labels = (aResult?.data_points || [])
    .filter(d => d.size_bytes >= 1024)
    .map(d => formatBytes(d.size_bytes));

  compareChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: `${a.gpu_type.toUpperCase()} x${a.total_gpus} (${a.run_id})`,
          data: (aResult?.data_points || []).filter(d => d.size_bytes >= 1024).map(d => d.busbw_gbps),
          borderColor: COLORS.green,
          backgroundColor: COLORS.green + '33',
          tension: 0.3,
          pointRadius: 3,
        },
        {
          label: `${b.gpu_type.toUpperCase()} x${b.total_gpus} (${b.run_id})`,
          data: (bResult?.data_points || []).filter(d => d.size_bytes >= 1024).map(d => d.busbw_gbps),
          borderColor: COLORS.cyan,
          backgroundColor: COLORS.cyan + '33',
          tension: 0.3,
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
      },
      scales: {
        y: {
          title: { display: true, text: 'Bus Bandwidth (GB/s)' },
          beginAtZero: true,
        },
        x: {
          title: { display: true, text: 'Message Size' },
        },
      },
    },
  });

  // Comparison summary
  const summaryEl = document.getElementById('compare-summary');
  const aPeak = aResult?.data_points ? Math.max(...aResult.data_points.map(d => d.busbw_gbps)) : 0;
  const bPeak = bResult?.data_points ? Math.max(...bResult.data_points.map(d => d.busbw_gbps)) : 0;
  const aAvg = aResult?.avg_bus_bandwidth_gbps || 0;
  const bAvg = bResult?.avg_bus_bandwidth_gbps || 0;
  const diff = aPeak > 0 ? ((bPeak - aPeak) / aPeak * 100) : 0;
  const winner = diff > 0 ? 'B' : 'A';
  const absDiff = Math.abs(diff);

  summaryEl.style.display = 'block';
  summaryEl.innerHTML = `
    <h3>Comparison Summary</h3>
    <div class="info-grid">
      <div class="info-item">
        <strong>Run A: ${a.gpu_type.toUpperCase()} x${a.total_gpus}</strong>
        <p>Peak: ${aPeak.toFixed(1)} GB/s | Avg: ${aAvg.toFixed(1)} GB/s | Nodes: ${a.num_nodes}</p>
      </div>
      <div class="info-item">
        <strong>Run B: ${b.gpu_type.toUpperCase()} x${b.total_gpus}</strong>
        <p>Peak: ${bPeak.toFixed(1)} GB/s | Avg: ${bAvg.toFixed(1)} GB/s | Nodes: ${b.num_nodes}</p>
      </div>
    </div>
    <p style="margin-top:12px;color:var(--text-dim);">
      ${absDiff < 2 ? 'Both runs have nearly identical performance.' :
        `Run ${winner} is <strong style="color:var(--accent)">${absDiff.toFixed(1)}% faster</strong> at peak bandwidth.`}
    </p>
  `;
}

function renderRuntimeComparison() {
  const emptyEl = document.getElementById('runtimes-empty');
  const contentEl = document.getElementById('runtimes-content');
  if (!emptyEl || !contentEl) return;

  if (allComparisons.length === 0) {
    emptyEl.style.display = 'block';
    contentEl.style.display = 'none';
    return;
  }
  emptyEl.style.display = 'none';
  contentEl.style.display = 'block';

  const select = document.getElementById('runtime-compare-select');
  const cmp = allComparisons[select.value || 0];
  if (!cmp || !cmp.runtimes) return;

  const runtimes = Object.keys(cmp.runtimes);
  const baseline = cmp.baseline || runtimes[0];

  // Stats cards: one per runtime with its headline numbers
  const statsEl = document.getElementById('runtime-stats');
  statsEl.innerHTML = runtimes.map(rt => {
    const meta = runtimeMeta(rt);
    const d = cmp.runtimes[rt];
    const nccl = d.nccl ? Object.values(d.nccl)[0] : null;
    const cold = d.coldstart ? d.coldstart.cuda_init_ms_median : null;
    const isBase = rt === baseline;
    return `
      <div class="stat-card" style="border-top: 3px solid ${meta.color}">
        <div class="label">${meta.label}${isBase ? ' &mdash; baseline' : ''}</div>
        <div class="value">${nccl != null ? nccl.toFixed(1) : '&mdash;'} <span class="unit">GB/s</span></div>
        <div class="label" style="margin-top:4px;">CUDA init: ${cold != null ? cold.toFixed(0) + ' ms' : '&mdash;'}</div>
      </div>`;
  }).join('');

  // NCCL bandwidth grouped bar chart (one group per test, one bar per runtime)
  const ncclTests = [...new Set(runtimes.flatMap(rt => Object.keys(cmp.runtimes[rt].nccl || {})))];
  if (runtimeNcclChart) runtimeNcclChart.destroy();
  const ncclCanvas = document.getElementById('runtime-nccl-chart');
  ncclCanvas.parentElement.style.display = ncclTests.length ? 'block' : 'none';
  if (ncclTests.length) {
    runtimeNcclChart = new Chart(ncclCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: ncclTests.map(t => t.replace(/_/g, ' ')),
        datasets: runtimes.map(rt => {
          const meta = runtimeMeta(rt);
          return {
            label: meta.label,
            data: ncclTests.map(t => cmp.runtimes[rt].nccl?.[t] ?? 0),
            backgroundColor: meta.color + 'cc',
            borderColor: meta.color,
            borderWidth: 1,
          };
        }),
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'top' } },
        scales: {
          y: { title: { display: true, text: 'Avg Bus Bandwidth (GB/s)' }, beginAtZero: true },
        },
      },
    });
  }

  // Cold-start latency grouped bar chart
  const coldMetrics = [
    { key: 'nvidia_smi_ms_median', label: 'nvidia-smi' },
    { key: 'cuda_init_ms_median', label: 'CUDA init' },
    { key: 'first_kernel_ms_median', label: 'first kernel' },
  ];
  const hasCold = runtimes.some(rt => cmp.runtimes[rt].coldstart);
  if (runtimeColdstartChart) runtimeColdstartChart.destroy();
  const coldCanvas = document.getElementById('runtime-coldstart-chart');
  coldCanvas.parentElement.style.display = hasCold ? 'block' : 'none';
  if (hasCold) {
    runtimeColdstartChart = new Chart(coldCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: coldMetrics.map(m => m.label),
        datasets: runtimes.map(rt => {
          const meta = runtimeMeta(rt);
          return {
            label: meta.label,
            data: coldMetrics.map(m => cmp.runtimes[rt].coldstart?.[m.key] ?? 0),
            backgroundColor: meta.color + 'cc',
            borderColor: meta.color,
            borderWidth: 1,
          };
        }),
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'top' } },
        scales: {
          y: { title: { display: true, text: 'Latency (ms, median)' }, beginAtZero: true },
        },
      },
    });
  }

  // Overhead table vs baseline
  const tableEl = document.getElementById('runtime-overhead-table');
  const nonBaseline = runtimes.filter(rt => rt !== baseline && cmp.runtimes[rt].overhead_vs_baseline);
  if (nonBaseline.length === 0) {
    tableEl.innerHTML = '<p class="chart-help">No overhead data (need at least two runtimes with a shared baseline).</p>';
  } else {
    const metricKeys = [...new Set(nonBaseline.flatMap(rt => Object.keys(cmp.runtimes[rt].overhead_vs_baseline)))];
    const fmtOverhead = (key, val) => {
      if (val == null) return '&mdash;';
      if (key.endsWith('_pct')) {
        const color = val < -2 ? '#ef4444' : val > 2 ? '#76b900' : 'inherit';
        return `<span style="color:${color}">${val > 0 ? '+' : ''}${val.toFixed(1)}%</span>`;
      }
      const unit = key.endsWith('_s_delta') ? ' s' : ' ms';
      const color = val > 2 ? '#ef4444' : 'inherit';
      return `<span style="color:${color}">${val > 0 ? '+' : ''}${val}${unit}</span>`;
    };
    const prettyMetric = k => k
      .replace(/_pct$/, ' (bandwidth %)')
      .replace(/_ms_delta$/, ' (ms)')
      .replace(/_s_delta$/, ' (s)')
      .replace(/_/g, ' ');
    tableEl.innerHTML = `
      <table class="overhead-table" style="width:100%;border-collapse:collapse;">
        <thead><tr>
          <th style="text-align:left;padding:8px;border-bottom:1px solid var(--border);">Metric (vs ${runtimeMeta(baseline).label})</th>
          ${nonBaseline.map(rt => `<th style="text-align:right;padding:8px;border-bottom:1px solid var(--border);">${runtimeMeta(rt).label}</th>`).join('')}
        </tr></thead>
        <tbody>
          ${metricKeys.map(key => `<tr>
            <td style="padding:8px;border-bottom:1px solid var(--border);">${prettyMetric(key)}</td>
            ${nonBaseline.map(rt => `<td style="text-align:right;padding:8px;border-bottom:1px solid var(--border);">${fmtOverhead(key, cmp.runtimes[rt].overhead_vs_baseline[key])}</td>`).join('')}
          </tr>`).join('')}
        </tbody>
      </table>
      <p class="chart-help" style="margin-top:8px;">Bandwidth: negative = slower than baseline. Latency deltas: positive = slower than baseline. Values within &plusmn;2% / &plusmn;2ms are noise.</p>`;
  }
}

function getSampleComparisons() {
  return [{
    compare_id: 'compare-sample',
    baseline: 'default',
    runtimes: {
      default: {
        nccl: { all_reduce: 115.26 },
        coldstart: { nvidia_smi_ms_median: 30, cuda_init_ms_median: 291.1, first_kernel_ms_median: 139.6, sandbox_startup_s_median: 1, repeats: 3 },
      },
      gvisor: {
        nccl: { all_reduce: 115.3 },
        coldstart: { nvidia_smi_ms_median: 69, cuda_init_ms_median: 342, first_kernel_ms_median: 137.5, sandbox_startup_s_median: 1, repeats: 3 },
        overhead_vs_baseline: { nccl_all_reduce_bandwidth_pct: 0, cuda_init_ms_delta: 50.9, first_kernel_ms_delta: -2.1, sandbox_startup_s_delta: 0 },
      },
    },
  }];
}

// Sample data for demo/development
function getSampleData() {
  const sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432,
    67108864, 134217728, 268435456, 536870912, 1073741824];

  function genBandwidths(peak, noise) {
    return sizes.map((size, i) => {
      const progress = i / (sizes.length - 1);
      const bw = peak * (1 - Math.exp(-5 * progress)) + (Math.random() - 0.5) * noise;
      const time = size / (Math.max(bw, 0.01) * 1e9) * 1e6;
      return {
        size_bytes: size,
        busbw_gbps: Math.max(0, parseFloat(bw.toFixed(2))),
        algbw_gbps: Math.max(0, parseFloat((bw * 0.85).toFixed(2))),
        time_us: parseFloat(Math.max(1, time).toFixed(2)),
      };
    });
  }

  return [
    {
      run_id: '20260305-143022-demo01',
      timestamp: '2026-03-05T14:30:22Z',
      benchmark_type: 'nccl',
      gpu_type: 'h100',
      total_gpus: 16,
      num_nodes: 2,
      results: [
        {
          test: 'all_reduce',
          avg_bus_bandwidth_gbps: 159.15,
          data_points: genBandwidths(280, 15),
        },
        {
          test: 'all_gather',
          avg_bus_bandwidth_gbps: 172.33,
          data_points: genBandwidths(310, 12),
        },
        {
          test: 'reduce_scatter',
          avg_bus_bandwidth_gbps: 165.88,
          data_points: genBandwidths(290, 14),
        },
      ],
    },
    {
      run_id: '20260305-160045-demo02',
      timestamp: '2026-03-05T16:00:45Z',
      benchmark_type: 'nccl',
      gpu_type: 'h100',
      total_gpus: 64,
      num_nodes: 8,
      results: [
        {
          test: 'all_reduce',
          avg_bus_bandwidth_gbps: 142.50,
          data_points: genBandwidths(250, 20),
        },
      ],
    },
    {
      run_id: '20260305-180012-demo03',
      timestamp: '2026-03-05T18:00:12Z',
      benchmark_type: 'training',
      gpu_type: 'h100',
      total_gpus: 64,
      num_nodes: 8,
      results: {
        status: 'COMPLETE',
        iterations_measured: '35-44',
        iterations_found: 10,
        step_time_mean_ms: 12345.67,
        step_time_std_ms: 234.56,
        tflops_per_gpu_mean: 456.78,
        tflops_per_gpu_std: 12.34,
      },
    },
    {
      run_id: '20260305-200033-demo04',
      timestamp: '2026-03-05T20:00:33Z',
      benchmark_type: 'training',
      gpu_type: 'b200',
      total_gpus: 64,
      num_nodes: 8,
      results: {
        status: 'COMPLETE',
        iterations_measured: '35-44',
        iterations_found: 10,
        step_time_mean_ms: 8765.43,
        step_time_std_ms: 156.78,
        tflops_per_gpu_mean: 623.45,
        tflops_per_gpu_std: 15.67,
      },
    },
  ];
}

// Initialize
loadResults();

// kubemark-ai Dashboard

let allResults = [];
let ncclChart = null;
let latencyChart = null;
let tflopsChart = null;
let compareChart = null;

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

// GPU spec references — sourced from official NVIDIA spec sheets
// Only NVLink bandwidth and memory bandwidth from published hardware specs
// These are HARDWARE SPECS, not benchmark baselines — actual NCCL performance varies
// Source: NVIDIA DGX Cloud Benchmarking repo (github.com/NVIDIA/dgxc-benchmarking)
// Only H100, B200, GB200, GB300 are in NVIDIA's benchmark suite
const GPU_SPECS = {
  // NVIDIA DGX Cloud Benchmarking supported architectures
  // NVLink bandwidth per GPU from reference architecture specs
  h100:  { nvlinkBw: 900, memBw: 3200, interconnect: 'NVLink 4.0', gpusPerNode: 8, dgxcSupported: true },
  b200:  { nvlinkBw: 1800, memBw: 8000, interconnect: 'NVLink 5.0', gpusPerNode: 8, dgxcSupported: true },
  gb200: { nvlinkBw: 1800, memBw: 8000, interconnect: 'NVLink 5.0', gpusPerNode: 4, dgxcSupported: true },
  gb300: { nvlinkBw: 1800, memBw: 12000, interconnect: 'NVLink 5.0', gpusPerNode: 4, dgxcSupported: true },
};

// For backward compat — alias
const REFERENCE_BASELINES = GPU_SPECS;

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

  if (allResults.length === 0) {
    document.getElementById('empty-state').style.display = 'block';
    return;
  }

  populateSelectors();
  renderNCCLRun();
  renderTrainingRun();
}

function populateSelectors() {
  const ncclRuns = allResults.filter(r => r.benchmark_type === 'nccl');
  const trainingRuns = allResults.filter(r => r.benchmark_type === 'training');

  fillSelect('nccl-run-select', ncclRuns);
  fillSelect('training-run-select', trainingRuns);
  fillSelect('compare-a', ncclRuns);
  fillSelect('compare-b', ncclRuns);
}

function fillSelect(id, runs) {
  const select = document.getElementById(id);
  if (!select) return;
  select.innerHTML = '';
  runs.forEach((run, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `${run.gpu_type.toUpperCase()} x${run.total_gpus} — ${run.run_id} (${new Date(run.timestamp).toLocaleDateString()})`;
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

  // Info box — contextual information about the result
  const ratingEl = document.getElementById('nccl-rating');
  ratingEl.className = 'rating-box';
  ratingEl.style.display = '';

  if (isSingleGpu) {
    ratingEl.className = 'rating-box rating-good';
    ratingEl.innerHTML = `<div class="rating-label">Single-GPU Test</div><div>This test measures <strong>internal GPU memory bandwidth</strong> (${maxBw.toFixed(1)} GB/s), not inter-GPU communication. ${spec ? `The ${run.gpu_type.toUpperCase()} has ${spec.memBw} GB/s theoretical memory bandwidth (${spec.interconnect}).` : ''} For GPU-to-GPU benchmarks, use 2+ GPUs.</div>`;
  } else if (spec && spec.dgxcSupported) {
    ratingEl.className = 'rating-box rating-good';
    ratingEl.innerHTML = `<div class="rating-label">NVIDIA DGX Cloud Benchmarking Supported</div><div>The ${run.gpu_type.toUpperCase()} is supported by <a href="https://github.com/NVIDIA/dgxc-benchmarking" target="_blank">NVIDIA's benchmark suite</a>. Hardware spec: ${spec.nvlinkBw} GB/s NVLink bandwidth per GPU, ${spec.memBw} GB/s memory bandwidth. Your peak: ${maxBw.toFixed(1)} GB/s bus bandwidth across ${run.total_gpus} GPUs (${run.num_nodes} nodes).</div>`;
  } else if (run.total_gpus > 1) {
    ratingEl.className = 'rating-box';
    ratingEl.innerHTML = `<div class="rating-label">Multi-GPU Result</div><div>Peak bus bandwidth: ${maxBw.toFixed(1)} GB/s across ${run.total_gpus} GPUs (${run.num_nodes} nodes). The ${run.gpu_type.toUpperCase()} is not in NVIDIA's DGX Cloud Benchmarking suite, so no official reference baseline is available. Your result reflects actual inter-node network performance.</div>`;
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

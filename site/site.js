// kubemark-ai landing site — renders published benchmark reports.
// The Pages workflow copies docs/benchmark-reports/*.json into ./reports/
// and generates ./reports/index.json listing the files.

const RUNTIME_LABELS = {
  default: 'runc (default)',
  runc: 'runc',
  gvisor: 'gVisor',
  vnode: 'vNode',
  'kata-qemu-nvidia-gpu': 'Kata (QEMU)',
};

const METRIC_LABELS = {
  nvidia_smi_ms_median: 'nvidia-smi (ms)',
  cuda_init_ms_median: 'CUDA init (ms)',
  first_kernel_ms_median: 'First kernel (ms)',
  sandbox_startup_s_median: 'Sandbox startup (s)',
};

function rtLabel(rt) { return RUNTIME_LABELS[rt] || rt; }

function fmtDelta(v, unit) {
  const cls = v > 0 ? 'delta-pos' : 'delta-neg';
  return `<span class="${cls}">${v > 0 ? '+' : ''}${v}${unit}</span>`;
}

function renderReport(report) {
  const runtimes = Object.keys(report.runtimes || {});
  if (!runtimes.length) return '';
  const base = report.baseline || runtimes[0];

  // Collect all coldstart metric keys present
  const csKeys = new Set();
  const ncclTests = new Set();
  runtimes.forEach(rt => {
    const e = report.runtimes[rt];
    Object.keys(e.coldstart || {}).forEach(k => { if (k !== 'repeats') csKeys.add(k); });
    Object.keys(e.nccl || {}).forEach(t => { if (t !== 'unknown') ncclTests.add(t); });
  });

  const rows = [];
  ncclTests.forEach(test => {
    const cells = runtimes.map(rt => {
      const v = report.runtimes[rt].nccl?.[test];
      if (v == null) return '<td>—</td>';
      const pct = report.runtimes[rt].overhead_vs_baseline?.[`nccl_${test}_bandwidth_pct`];
      return `<td>${v} GB/s${pct != null && rt !== base ? ` (${fmtDelta(pct, '%')})` : ''}</td>`;
    });
    rows.push(`<tr><td>NCCL ${test} busbw</td>${cells.join('')}</tr>`);
  });
  csKeys.forEach(key => {
    const unit = key.endsWith('_s_median') ? ' s' : ' ms';
    const deltaKey = key.replace('_median', '_delta');
    const cells = runtimes.map(rt => {
      const v = report.runtimes[rt].coldstart?.[key];
      if (v == null) return '<td>—</td>';
      const d = report.runtimes[rt].overhead_vs_baseline?.[deltaKey];
      return `<td>${v}${unit}${d != null && rt !== base ? ` (${fmtDelta(d, unit.trim())})` : ''}</td>`;
    });
    rows.push(`<tr><td>${METRIC_LABELS[key] || key}</td>${cells.join('')}</tr>`);
  });

  const repeats = Math.max(...runtimes.map(rt => report.runtimes[rt].coldstart?.repeats || 1));
  return `
    <div class="report">
      <h3>${report.compare_id}${report.vcluster ? '<span class="vc-badge">inside vCluster</span>' : ''}</h3>
      <div class="report-meta">baseline: ${rtLabel(base)} · medians of ${repeats} repeats</div>
      <table>
        <thead><tr><th>Metric</th>${runtimes.map(rt => `<th>${rtLabel(rt)}</th>`).join('')}</tr></thead>
        <tbody>${rows.join('')}</tbody>
      </table>
    </div>`;
}

async function loadReports() {
  const container = document.getElementById('reports');
  try {
    const idx = await (await fetch('reports/index.json')).json();
    const reports = await Promise.all(
      idx.map(f => fetch(`reports/${f}`).then(r => r.json()).catch(() => null))
    );
    const html = reports.filter(Boolean).reverse().map(renderReport).join('');
    container.innerHTML = html || '<p class="section-sub">No published reports yet.</p>';
  } catch (e) {
    container.innerHTML = '<p class="section-sub">Reports are published from <a href="https://github.com/saiyam1814/kubemark-ai/tree/main/docs/benchmark-reports">docs/benchmark-reports</a>.</p>';
  }
}

loadReports();

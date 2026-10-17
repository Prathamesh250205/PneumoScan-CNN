const $ = (id) => document.getElementById(id);
const drop = $('drop'), fileInput = $('file'), preview = $('preview'), dropEmpty = $('drop-empty');
const analyzeBtn = $('analyze'), clearBtn = $('clear'), fileName = $('file-name');
const resultEmpty = $('result-empty'), resultBody = $('result-body'), errorBox = $('error');
const status = $('status');

let upload = null; // Blob sent to /predict

// --- Server status ---
async function checkHealth() {
    try {
        const res = await fetch('/health');
        const data = await res.json();
        if (data.model_loaded) setStatus('ok', 'Online · 2 models');
        else setStatus('warn', 'Model not loaded');
    } catch {
        setStatus('down', 'Server offline');
    }
}
function setStatus(state, text) {
    status.dataset.state = state;
    status.querySelector('.status-text').textContent = text;
}
checkHealth();

// --- File selection ---
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => e.dataTransfer.files[0] && selectFile(e.dataTransfer.files[0]));
fileInput.addEventListener('change', () => fileInput.files[0] && selectFile(fileInput.files[0]));
clearBtn.addEventListener('click', reset);
document.querySelectorAll('[data-sample]').forEach((btn) => btn.addEventListener('click', async () => {
    const name = btn.dataset.sample;
    const blob = await (await fetch(`/samples/${name}.jpg`)).blob();
    await selectFile(new File([blob], `sample-${name}.jpg`, { type: 'image/jpeg' }));
    analyzeBtn.click();
}));

async function selectFile(file) {
    hideError();
    if (!file.type.startsWith('image/')) return showError('That file is not an image. Use PNG, JPEG or WebP.');
    try {
        upload = await downscale(file, 1024);
    } catch {
        return showError('Could not read that image.');
    }
    preview.src = URL.createObjectURL(upload);
    preview.hidden = false;
    dropEmpty.hidden = true;
    drop.classList.add('has-file');
    fileName.textContent = file.name;
    clearBtn.hidden = false;
    analyzeBtn.disabled = false;
    showResult(false);
}

// The model sees 224x224 anyway; shrinking client-side keeps uploads under serverless body limits (4.5 MB on Vercel).
async function downscale(file, maxSide) {
    const bmp = await createImageBitmap(file);
    const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bmp.width * scale);
    canvas.height = Math.round(bmp.height * scale);
    canvas.getContext('2d').drawImage(bmp, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
}

function reset() {
    upload = null;
    fileInput.value = '';
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.removeAttribute('src');
    preview.hidden = true;
    dropEmpty.hidden = false;
    drop.classList.remove('has-file');
    fileName.textContent = 'No file selected';
    clearBtn.hidden = true;
    analyzeBtn.disabled = true;
    hideError();
    showResult(false);
}

// --- Inference ---
analyzeBtn.addEventListener('click', async () => {
    if (!upload) return;
    hideError();
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing…';
    drop.classList.add('scanning');
    const form = new FormData();
    form.append('file', upload, 'xray.jpg');
    const t0 = performance.now();
    try {
        const res = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
        render(data, Math.round(performance.now() - t0));
    } catch (err) {
        showError(`${err.message}. The server may be waking up, so try again in a few seconds.`);
    } finally {
        drop.classList.remove('scanning');
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze';
    }
});

// 0.946: highest ResNet18 threshold keeping 95% sensitivity on the test set (scripts/export_onnx.py)
const LEVELS = [
    { min: 0.946, level: 'high', verdict: 'Pneumonia likely', note: 'Above the 0.946 screening threshold, which catches 95% of pneumonia cases in the test set.' },
    { min: 0.5, level: 'mid', verdict: 'Borderline', note: 'The model leans pneumonia, but not strongly enough to pass the screening threshold.' },
    { min: 0, level: 'low', verdict: 'Normal likely', note: 'Below 0.5. The model favours a normal X-ray.' },
];

function render({ pneumonia_probability: p, normal_probability: n, models }, ms) {
    for (const [name, m] of Object.entries(models)) {
        const img = $(`cam-${name}`);
        img.classList.remove('wipe');
        img.src = m.heatmap;
        void img.offsetWidth; // restart the reveal animation on every result
        img.classList.add('wipe');
        $(`cam-p-${name}`).textContent = m.pneumonia_probability.toFixed(3);
    }
    $('cams').hidden = false;
    const { level, verdict, note } = LEVELS.find((l) => p >= l.min);
    const v = $('verdict');
    v.textContent = verdict;
    v.dataset.level = level;
    $('verdict-note').textContent = note;
    $('p-pneu').textContent = p.toFixed(3);
    $('p-norm').textContent = n.toFixed(3);
    $('latency').textContent = `${ms} ms`;
    $('meter-marker').style.left = `${(p * 100).toFixed(1)}%`;
    $('meter').setAttribute('aria-label', `Pneumonia probability ${p.toFixed(2)} on a 0 to 1 scale`);
    showResult(true);
    if (window.innerWidth < 900) $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showResult(on) { resultBody.hidden = !on; resultEmpty.hidden = on; if (!on) $('cams').hidden = true; }
function showError(msg) { errorBox.textContent = msg; errorBox.hidden = false; }
function hideError() { errorBox.hidden = true; }

// --- Motion: exploded hero, scroll reveals, count-ups ---
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
const stage = $('stage');
const stack = stage.querySelector('.stack');

requestAnimationFrame(() => setTimeout(() => stage.classList.add('exploded'), calm ? 0 : 350));

if (!calm && matchMedia('(pointer: fine)').matches) {
    let frame = 0;
    stage.addEventListener('pointermove', (e) => {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => {
            const r = stage.getBoundingClientRect();
            const x = (e.clientX - r.left) / r.width - 0.5;
            const y = (e.clientY - r.top) / r.height - 0.5;
            stack.style.setProperty('--tz', `${(x * 14).toFixed(2)}deg`);
            stack.style.setProperty('--tx', `${(-y * 10).toFixed(2)}deg`);
        });
    });
    stage.addEventListener('pointerleave', () => { stack.style.setProperty('--tz', '0deg'); stack.style.setProperty('--tx', '0deg'); });
}

// Hover or focus a legend entry to isolate that layer
stage.querySelectorAll('.legend button').forEach((btn) => {
    const on = () => {
        stage.dataset.active = btn.dataset.layer;
        stage.querySelectorAll('.legend button').forEach((b) => b.classList.toggle('on', b === btn));
    };
    const off = () => { delete stage.dataset.active; btn.classList.remove('on'); };
    btn.addEventListener('pointerenter', on);
    btn.addEventListener('focus', on);
    btn.addEventListener('pointerleave', off);
    btn.addEventListener('blur', off);
});

function countUp(el) {
    const target = Number(el.dataset.count);
    const unit = el.querySelector('small')?.outerHTML ?? '';
    const t0 = performance.now();
    const tick = (t) => {
        const k = Math.min(1, (t - t0) / 900);
        el.innerHTML = Math.round(target * (1 - (1 - k) ** 3)) + unit;
        if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}

const revealables = document.querySelectorAll('.section-head, .prose, .table-scroll, .figure, .steps li, .limits li');
if (calm || !('IntersectionObserver' in window)) {
    document.querySelectorAll('.dotplot').forEach((el) => el.classList.add('in'));
} else {
    revealables.forEach((el) => el.classList.add('reveal'));
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
        if (!e.isIntersecting) return;
        e.target.classList.add('in');
        if (e.target.matches('.stats')) e.target.querySelectorAll('[data-count]').forEach(countUp);
        io.unobserve(e.target);
    }), { rootMargin: '0px 0px -12% 0px' });
    [...revealables, ...document.querySelectorAll('.dotplot, .stats')].forEach((el) => io.observe(el));
}

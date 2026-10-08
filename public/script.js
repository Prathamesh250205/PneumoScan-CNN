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
        if (!data.model_loaded) setStatus('warn', 'Model not loaded');
        else if (data.placeholder_weights) setStatus('warn', 'Online · demo weights');
        else setStatus('ok', 'Online · ResNet18');
        $('placeholder-notice').hidden = !data.placeholder_weights;
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
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze';
    }
});

const LEVELS = [
    { min: 0.70, level: 'high', verdict: 'Pneumonia likely', note: 'Output is above the 0.70 screening threshold. A clinician would need to review the image.' },
    { min: 0.35, level: 'mid', verdict: 'Indeterminate', note: 'Output falls in the 0.35–0.70 band, which this demo treats as uncertain.' },
    { min: 0, level: 'low', verdict: 'Normal likely', note: 'Output is below the 0.35 threshold. No consolidation pattern detected by the model.' },
];

function render({ pneumonia_probability: p, normal_probability: n }, ms) {
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

function showResult(on) { resultBody.hidden = !on; resultEmpty.hidden = on; }
function showError(msg) { errorBox.textContent = msg; errorBox.hidden = false; }
function hideError() { errorBox.hidden = true; }

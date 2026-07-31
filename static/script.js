// Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPrompt = document.getElementById('upload-prompt');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsContainer = document.getElementById('results-container');
const systemStatus = document.getElementById('system-status');

// Metrics elements
const inferenceTimeVal = document.getElementById('inference-time');
const gaugePercentage = document.getElementById('gauge-percentage');
const gaugeFill = document.getElementById('gauge-fill');
const gaugeCaption = document.getElementById('gauge-caption');
const normalValue = document.getElementById('normal-value');
const normalProgress = document.getElementById('normal-progress');
const pneumoniaValue = document.getElementById('pneumonia-value');
const pneumoniaProgress = document.getElementById('pneumonia-progress');

// Diagnosis elements
const diagnosisCard = document.getElementById('diagnosis-card');
const diagnosisIcon = document.getElementById('diagnosis-icon');
const diagnosisTitle = document.getElementById('diagnosis-title');
const diagnosisDesc = document.getElementById('diagnosis-desc');

let selectedFile = null;

// Initialization: Check Backend Health
async function checkHealth() {
    try {
        const response = await fetch('/health');
        if (!response.ok) throw new Error('Degraded service');
        
        const data = await response.json();
        
        if (data.status === 'healthy') {
            updateStatus('healthy', 'Server Online • Model Loaded');
        } else {
            updateStatus('warning', 'Server Online • Missing Model');
            console.warn("FastAPI backend report: ONNX model not found in backend/ directory.");
        }
    } catch (error) {
        updateStatus('danger', 'Server Offline • Reconnecting...');
        console.error('Health check failed:', error);
    }
}

function updateStatus(state, message) {
    const dot = systemStatus.querySelector('.status-dot');
    const text = systemStatus.querySelector('.status-text');
    
    dot.className = `status-dot ${state}`;
    text.textContent = message;
}

// Run health check on load, and poll every 10 seconds
checkHealth();
setInterval(checkHealth, 10000);

// Tab switching logic
window.switchTab = function(tabId) {
    // Deactivate all tabs and contents
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Find matching button using its click target context or text match
    const eventTarget = window.event ? window.event.currentTarget : null;
    if (eventTarget) {
        eventTarget.classList.add('active');
    } else {
        // Fallback fallback selector
        const buttons = document.querySelectorAll('.tab-btn');
        if (tabId === 'tab-output') buttons[0].classList.add('active');
        if (tabId === 'tab-pneumonia') buttons[1].classList.add('active');
        if (tabId === 'tab-normal') buttons[2].classList.add('active');
    }
    
    // Activate content
    document.getElementById(tabId).classList.add('active');
};

// Drag and drop event listeners
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

dropZone.addEventListener('click', () => {
    // Only trigger file picker if clicking when prompt is visible
    if (!previewContainer.classList.contains('hidden')) return;
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Stop click from triggering parent file input click
    clearFile();
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (PNG, JPEG, etc.)');
        return;
    }
    
    // 10MB limit
    if (file.size > 10 * 1024 * 1024) {
        alert('File size exceeds 10MB limit. Please upload a smaller image.');
        return;
    }

    selectedFile = file;
    
    // Render preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadPrompt.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '#';
    previewContainer.classList.add('hidden');
    uploadPrompt.classList.remove('hidden');
    analyzeBtn.disabled = true;
    resultsContainer.classList.add('hidden');
    resetMetrics();
}

function resetMetrics() {
    inferenceTimeVal.textContent = '--';
    gaugePercentage.textContent = '0%';
    gaugeFill.style.background = `conic-gradient(
        rgba(255, 255, 255, 0.08) 0deg,
        rgba(255, 255, 255, 0.08) 180deg,
        transparent 180deg
    )`;
    gaugeCaption.textContent = '--';
    normalValue.textContent = '0%';
    normalProgress.style.width = '0%';
    pneumoniaValue.textContent = '0%';
    pneumoniaProgress.style.width = '0%';
}

// API prediction handler
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Image...';
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    const startTime = performance.now();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const duration = Math.round(performance.now() - startTime);
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Prediction failed');
        }
        
        const data = await response.json();
        renderResults(data, duration);
        
    } catch (error) {
        console.error('Prediction Error:', error);
        alert(`Inference Error: ${error.message}\n\nNote: If deploying for the first time, make sure model.onnx is committed inside the "backend" directory.`);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-microscope"></i> Run CNN Classification';
    }
});

function renderResults(data, latency) {
    const normalProb = data.normal_probability;
    const pneumoniaProb = data.pneumonia_probability;
    
    // Display container
    resultsContainer.classList.remove('hidden');
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Set inference metadata
    inferenceTimeVal.textContent = latency;
    
    // Calculate rounded percentage values
    const normalPct = Math.round(normalProb * 100);
    const pneumoniaPct = Math.round(pneumoniaProb * 100);
    
    // Update numerical lists and bars
    normalValue.textContent = `${normalPct}%`;
    normalProgress.style.width = `${normalPct}%`;
    
    pneumoniaValue.textContent = `${pneumoniaPct}%`;
    pneumoniaProgress.style.width = `${pneumoniaPct}%`;
    
    // Update Semi-circular gauge
    // Sweeps from 0 to 180 degrees based on pneumonia probability
    const deg = Math.round(pneumoniaProb * 180);
    gaugePercentage.textContent = `${pneumoniaPct}%`;
    
    // Dynamic color coding based on threshold risk
    let themeColor;
    if (pneumoniaProb >= 0.70) {
        themeColor = 'var(--color-pneumonia)'; // High Coral risk
        gaugeCaption.textContent = 'High probability of Pneumonia';
        gaugeCaption.style.color = 'var(--color-pneumonia)';
    } else if (pneumoniaProb >= 0.35) {
        themeColor = 'var(--color-warning)';   // Intermediate Amber risk
        gaugeCaption.textContent = 'Indeterminate screening status';
        gaugeCaption.style.color = 'var(--color-warning)';
    } else {
        themeColor = 'var(--color-normal)';    // Low Teal risk
        gaugeCaption.textContent = 'Consistent with Normal X-ray';
        gaugeCaption.style.color = 'var(--color-normal)';
    }
    
    // Paint conically
    gaugeFill.style.background = `conic-gradient(
        ${themeColor} 0deg,
        ${themeColor} ${deg}deg,
        rgba(255, 255, 255, 0.08) ${deg}deg,
        rgba(255, 255, 255, 0.08) 180deg,
        transparent 180deg
    )`;
    
    // Render Clinical Diagnostic Assessment Card
    updateDiagnosisCard(pneumoniaProb);
}

function updateDiagnosisCard(pneumoniaProb) {
    diagnosisCard.className = 'diagnosis-card'; // Clear states
    
    if (pneumoniaProb >= 0.70) {
        diagnosisCard.classList.add('pneumonia-active');
        diagnosisIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation" style="color: var(--color-pneumonia);"></i>';
        diagnosisTitle.textContent = 'Risk Category: Elevated Risk';
        diagnosisDesc.textContent = 'Model indicates localized opacities or signs consistent with focal infection. Clinical verification and correlation with patient symptoms are required.';
        
        // CSS Style Injection for customized theme colors in diagnosis box
        diagnosisCard.style.borderLeft = '4px solid var(--color-pneumonia)';
        diagnosisCard.style.backgroundColor = 'var(--color-pneumonia-bg)';
    } else if (pneumoniaProb >= 0.35) {
        diagnosisCard.classList.add('warning-active');
        diagnosisIcon.innerHTML = '<i class="fa-solid fa-circle-info" style="color: var(--color-warning);"></i>';
        diagnosisTitle.textContent = 'Risk Category: Borderline/Indeterminate';
        diagnosisDesc.textContent = 'The classifier output lies in a moderate-risk range. Recommend secondary screening or detailed inspection of radiological features.';
        
        diagnosisCard.style.borderLeft = '4px solid var(--color-warning)';
        diagnosisCard.style.backgroundColor = 'rgba(230, 160, 0, 0.06)';
    } else {
        diagnosisCard.classList.add('normal-active');
        diagnosisIcon.innerHTML = '<i class="fa-solid fa-circle-check" style="color: var(--color-normal);"></i>';
        diagnosisTitle.textContent = 'Risk Category: Unremarkable Screening';
        diagnosisDesc.textContent = 'X-ray scan does not demonstrate significant focal consolidation anomalies. Findings suggest normal pulmonary fields.';
        
        diagnosisCard.style.borderLeft = '4px solid var(--color-normal)';
        diagnosisCard.style.backgroundColor = 'var(--color-normal-bg)';
    }
}

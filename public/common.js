const $ = (id) => document.getElementById(id);
const status = $('status');

// --- Theme toggle: explicit choice wins over the OS setting, remembered per visitor ---
const root = document.documentElement;
const themeBtn = $('theme-toggle');
const isDark = () => (root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches);
const labelTheme = () => themeBtn.setAttribute('aria-label', isDark() ? 'Switch to light theme' : 'Switch to dark theme');
themeBtn.addEventListener('click', () => {
    root.dataset.theme = isDark() ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch {}
    labelTheme();
});
labelTheme();

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

// --- Motion: exploded hero, scroll reveals, count-ups ---
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
const stage = $('stage');
const stack = stage?.querySelector('.stack');

if (stage) requestAnimationFrame(() => setTimeout(() => stage.classList.add('exploded'), calm ? 0 : 350));

// Idle motion pauses while the hero is off screen
if (stage && 'IntersectionObserver' in window) {
    new IntersectionObserver(([e]) => stage.classList.toggle('paused', !e.isIntersecting)).observe(stage);
}

if (stage && !calm && matchMedia('(pointer: fine)').matches) {
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
stage?.querySelectorAll('.legend button').forEach((btn) => {
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

// Attention landscape: the hero X-ray as a 3D surface whose height is a model's Grad-CAM.
// Layers come from scripts/make_hero_layers.py; attention intensity is stored in the PNG alpha channel.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const host = document.getElementById('land');
const buttons = [...document.querySelectorAll('.land-toggle button')];
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
const SEG_X = 160, SEG_Y = 120, AMP = 1.25; // grid resolution; peak height in scene units (plane is 4 x 3)

const load = (src) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
});

function canvasOf(w, h, ...imgs) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    imgs.forEach((img) => ctx.drawImage(img, 0, 0, w, h));
    return c;
}

// Height per vertex from the attention layer's alpha, smoothed by the downsampling itself
function heightsFrom(img) {
    const data = canvasOf(SEG_X + 1, SEG_Y + 1, img).getContext('2d').getImageData(0, 0, SEG_X + 1, SEG_Y + 1).data;
    const h = new Float32Array((SEG_X + 1) * (SEG_Y + 1));
    for (let i = 0; i < h.length; i++) h[i] = (data[i * 4 + 3] / 255) * AMP;
    return h;
}

async function main() {
    const [xray, lungs, ...cams] = await Promise.all(
        ['xray', 'lungs', 'cam_resnet18', 'cam_mobilenet_v2'].map((n) => load(`/images/hero/${n}.webp`)));
    const models = ['resnet18', 'mobilenet_v2'];
    const layers = Object.fromEntries(models.map((m, i) => {
        const tex = new THREE.CanvasTexture(canvasOf(1024, 768, xray, cams[i], lungs));
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = 4;
        return [m, { tex, heights: heightsFrom(cams[i]) }];
    }));

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50);
    camera.up.set(0, 0, 1);
    camera.position.set(0.5, -5.8, 3.1);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x30343c, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(-3, -2, 5);
    scene.add(sun);

    const geometry = new THREE.PlaneGeometry(4, 3, SEG_X, SEG_Y);
    const pos = geometry.attributes.position;
    const material = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableZoom = false;
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.minPolarAngle = 0.15;
    controls.maxPolarAngle = 1.25;
    controls.autoRotate = !calm;
    controls.autoRotateSpeed = 0.5;

    const cur = new Float32Array(pos.count); // heights currently on screen
    let from = cur.slice(), to = cur, t0 = 0, done = true, active = null;
    const MORPH_MS = calm ? 1 : 900;

    function show(model) {
        if (model === active) return;
        active = model;
        from = cur.slice();
        to = layers[model].heights;
        t0 = performance.now();
        done = false;
        material.map = layers[model].tex;
        material.needsUpdate = true;
        buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.model === model)));
    }

    // PlaneGeometry vertices run row by row from the top edge, matching image rows
    function applyHeights(now) {
        if (done) return;
        const k = Math.min(1, (now - t0) / MORPH_MS);
        const e = 1 - (1 - k) ** 3;
        for (let i = 0; i < pos.count; i++) pos.setZ(i, (cur[i] = from[i] + (to[i] - from[i]) * e));
        pos.needsUpdate = true;
        geometry.computeVertexNormals();
        done = k >= 1;
    }

    // Until the visitor picks one, alternate between the models so the peaks visibly move
    let userPicked = false;
    buttons.forEach((b) => b.addEventListener('click', () => { userPicked = true; show(b.dataset.model); }));
    renderer.domElement.addEventListener('pointerdown', () => { userPicked = true; });
    if (!calm) setInterval(() => { if (!userPicked && visible) show(active === 'resnet18' ? 'mobilenet_v2' : 'resnet18'); }, 5000);

    function resize() {
        const { width, height } = host.getBoundingClientRect();
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    }
    new ResizeObserver(resize).observe(host);
    resize();

    let visible = true;
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(host);

    show('resnet18');
    host.classList.add('ready');
    renderer.setAnimationLoop((now) => {
        if (!visible || document.hidden) return;
        applyHeights(now);
        controls.update();
        renderer.render(scene, camera);
    });
}

main().catch((err) => {
    console.warn('Attention landscape unavailable, showing the flat image instead.', err);
    host.classList.add('failed');
    document.querySelector('.land-hint')?.setAttribute('hidden', '');
});

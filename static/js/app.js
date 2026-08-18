/**
 * ═══════════════════════════════════════════════════════════════════
 * NEXUS FALLNET AI v5 — Optimized Cinematic UI App Logic
 * Scroll-driven frame animation, floating particles, glassmorphism,
 * and all core detection functionalities.
 * ═══════════════════════════════════════════════════════════════════
 */

// ═══════════════════════════════════════════════════════════════════
// 0. HUD COUNTER ANIMATION
// ═══════════════════════════════════════════════════════════════════
(function initHudCounters() {
    function animateCounter(el, target, duration, suffix) {
        if (!el) return;
        let start = null;
        function step(ts) {
            if (!start) start = ts;
            const progress = Math.min((ts - start) / duration, 1);
            // Ease out
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.floor(eased * target) + (suffix || '');
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    // Trigger when hero is visible
    const hero = document.getElementById('hero');
    if (!hero) return;
    const observer = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) {
            // Delay slightly for visual polish
            setTimeout(() => {
                animateCounter(document.getElementById('hud-c1'), 300, 1800);
                animateCounter(document.getElementById('hud-c2'), 156, 2000);
                animateCounter(document.getElementById('hud-c3'), 17, 900);
            }, 600);
            observer.disconnect();
        }
    }, { threshold: 0.3 });
    observer.observe(hero);
})();

// ═══════════════════════════════════════════════════════════════════
// 1. FLOATING PARTICLE SYSTEM
// ═══════════════════════════════════════════════════════════════════
(function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width, height;
    let particles = [];

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.size = Math.random() * 2 + 0.5;
            this.speedX = Math.random() * 0.5 - 0.25;
            this.speedY = Math.random() * 0.5 - 0.25;
            this.opacity = Math.random() * 0.5 + 0.1;
        }
        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            if (this.x < 0) this.x = width;
            if (this.x > width) this.x = 0;
            if (this.y < 0) this.y = height;
            if (this.y > height) this.y = 0;
        }
        draw() {
            ctx.fillStyle = `rgba(0, 242, 254, ${this.opacity})`;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function init() {
        particles = [];
        // Reduced from 100 to 45 — imperceptible visual diff, big GPU saving
        const numParticles = Math.min(Math.floor(window.innerWidth / 28), 45);
        for (let i = 0; i < numParticles; i++) {
            particles.push(new Particle());
        }
    }

    let rafTick = 0;
    function animate() {
        rafTick++;
        // Throttle: only draw every 2nd frame (~30fps instead of 60fps)
        if (rafTick % 2 === 0) {
            ctx.clearRect(0, 0, width, height);
            particles.forEach(p => {
                p.update();
                p.draw();
            });
        }
        requestAnimationFrame(animate);
    }

    init();
    animate();
})();


// ═══════════════════════════════════════════════════════════════════
// 2. SCROLL-DRIVEN SEQUENCE ANIMATION
// ═══════════════════════════════════════════════════════════════════
(function initScrollSequence() {
    const canvas = document.getElementById('sequence-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const frameFolder = '/static/animation1';
    const frameCount = 11;
    const images = [];
    let imagesLoaded = 0;
    let currentFrameIndex = 1;
    let targetFrameIndex = 1;

    // Preload images (using PNG format as copied by agent)
    for (let i = 1; i <= frameCount; i++) {
        const img = new Image();
        const paddedIndex = String(i).padStart(3, '0');
        img.onload = () => {
            imagesLoaded++;
            if (imagesLoaded === 1) resizeCanvas();
            if (imagesLoaded === frameCount) resizeCanvas();
        };
        img.src = `${frameFolder}/frame_${paddedIndex}.png`;
        images.push(img);
    }

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        renderFrame(Math.round(currentFrameIndex));
    }
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('load', resizeCanvas);

    function renderFrame(index) {
        if (index < 1 || index > frameCount) return;
        const img = images[index - 1];
        if (!img || !img.complete) return;

        const canvasAspect = canvas.width / canvas.height;
        const imgAspect = img.width / img.height;
        let drawW, drawH, offX = 0, offY = 0;

        if (canvasAspect > imgAspect) {
            drawW = canvas.width;
            drawH = canvas.width / imgAspect;
            offY = (canvas.height - drawH) / 2;
        } else {
            drawH = canvas.height;
            drawW = canvas.height * imgAspect;
            offX = (canvas.width - drawW) / 2;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.globalCompositeOperation = 'source-over';
        ctx.drawImage(img, offX, offY, drawW, drawH);
    }

    function tick() {
        if (Math.round(currentFrameIndex) !== targetFrameIndex) {
            const diff = targetFrameIndex - currentFrameIndex;
            currentFrameIndex += diff * 0.15; // Smooth interpolation
            const idx = Math.round(currentFrameIndex);
            renderFrame(Math.min(frameCount, Math.max(1, idx)));
        }
        requestAnimationFrame(tick);
    }
    tick();

    // Map scroll to frame
    window.addEventListener('scroll', () => {
        const scrollTop = window.scrollY || document.documentElement.scrollTop;
        // Animation plays over the first 1.5x viewport height (Hero + Problem section)
        const maxScroll = window.innerHeight * 1.5; 
        const frac = Math.min(Math.max(scrollTop / maxScroll, 0), 1);
        const frameIndex = Math.min(frameCount, Math.max(1, Math.floor(frac * (frameCount - 1)) + 1));
        
        if (frameIndex !== targetFrameIndex) {
            targetFrameIndex = frameIndex;
        }
    }, { passive: true });
})();


// ═══════════════════════════════════════════════════════════════════
// 3. NAVBAR + SCROLL TRACKING
// ═══════════════════════════════════════════════════════════════════
(function() {
    const nav = document.getElementById('navbar');
    const links = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');
    
    window.addEventListener('scroll', () => {
        nav.classList.toggle('scrolled', window.scrollY > 50);
        
        let currentId = '';
        sections.forEach(s => { 
            if (window.scrollY >= (s.offsetTop - 250)) {
                currentId = s.id; 
            }
        });
        
        links.forEach(l => {
            l.classList.toggle('active', l.getAttribute('href') === '#' + currentId);
        });
    });
})();


// ═══════════════════════════════════════════════════════════════════
// 4. 3D SCROLL REVEAL (Intersection Observer)
// ═══════════════════════════════════════════════════════════════════
(function() {
    const observer = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                e.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    
    const targets = document.querySelectorAll('.story-3d, .fc-3d, .ps-3d, .model-dcard, .conf-card, .tech-card');
    targets.forEach(el => {
        el.classList.add('reveal-3d');
        observer.observe(el);
    });
})();


// ═══════════════════════════════════════════════════════════════════
// 5. MODEL COMPARISON DASHBOARD
// ═══════════════════════════════════════════════════════════════════
let benchmarkData = null;

(async function loadModels() {
    try {
        const resp = await fetch('/api/benchmark');
        benchmarkData = await resp.json();
        renderComparison(benchmarkData);
        renderModelCards(benchmarkData);
        renderConfusion(benchmarkData);
        renderModelPickers(benchmarkData);
    } catch (e) {
        console.error('Failed to load benchmark data:', e);
    }
})();

function renderComparison(data) {
    const metrics = ['recall', 'precision', 'f1', 'fps'];
    const labels = ['Recall', 'Precision', 'F1 Score', 'Speed (FPS)'];
    const maxVals = [100, 100, 100, 40];
    const grid = document.getElementById('comparison-grid');
    if (!grid) return;
    grid.innerHTML = '';

    metrics.forEach((m, idx) => {
        const div = document.createElement('div');
        div.className = 'comp-metric';
        let bars = '';
        
        data.models.forEach(model => {
            const val = model.metrics[m];
            const pct = Math.min(100, (val / maxVals[idx]) * 100);
            const cls = model.generation === 'v8' ? 'v8' : model.generation === 'v11' ? 'v11' : 'v26';
            const displayName = model.name.replace('-Pose', '');
            bars += `<div class="comp-bar-row"><span class="comp-bar-label">${displayName}</span><div class="comp-bar-track"><div class="comp-bar-fill ${cls}" style="width:0%" data-width="${pct}%">${m==='fps' ? val.toFixed(1) : val.toFixed(1)+'%'}</div></div></div>`;
        });
        
        div.innerHTML = `<div class="comp-metric-label">${labels[idx]}</div><div class="comp-bars">${bars}</div>`;
        grid.appendChild(div);
    });

    const barObserver = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                e.target.querySelectorAll('.comp-bar-fill').forEach(bar => { bar.style.width = bar.dataset.width; });
                barObserver.unobserve(e.target);
            }
        });
    }, { threshold: 0.3 });
    barObserver.observe(grid);
}

function renderModelCards(data) {
    const container = document.getElementById('model-detail-cards');
    if (!container) return;
    container.innerHTML = '';
    
    data.models.forEach(model => {
        const isChamp = model.badge.includes('Champion');
        const card = document.createElement('div');
        card.className = 'model-dcard' + (isChamp ? ' champion' : '');
        card.innerHTML = `
            <div class="model-dcard-header">
                <div class="model-dcard-name">${model.name}</div>
                <div class="model-badge${isChamp ? ' champ' : ''}">${model.badge}</div>
            </div>
            <div class="model-dcard-desc">${model.description}</div>
            <div class="model-metrics-mini">
                <div class="mm-item"><div class="mm-label">Recall</div><div class="mm-val">${model.metrics.recall.toFixed(1)}%</div></div>
                <div class="mm-item"><div class="mm-label">Precision</div><div class="mm-val">${model.metrics.precision.toFixed(1)}%</div></div>
                <div class="mm-item"><div class="mm-label">F1 Score</div><div class="mm-val">${model.metrics.f1.toFixed(1)}%</div></div>
                <div class="mm-item"><div class="mm-label">Speed</div><div class="mm-val">${model.metrics.fps.toFixed(1)}</div></div>
            </div>
            <div class="model-strengths"><h4>Strengths</h4>${model.strengths.map(s=>`<div class="str-item">${s}</div>`).join('')}</div>
            <div class="model-strengths" style="margin-top:8px"><h4>Trade-offs</h4>${model.weaknesses.map(w=>`<div class="weak-item">${w}</div>`).join('')}</div>
        `;
        container.appendChild(card);
    });
}

function renderConfusion(data) {
    const grid = document.getElementById('confusion-grid');
    if (!grid) return;
    grid.innerHTML = '';
    
    data.models.forEach(model => {
        const c = model.confusion;
        const card = document.createElement('div');
        card.className = 'conf-card';
        card.innerHTML = `
            <div class="conf-card-title">${model.name}</div>
            <div class="conf-matrix">
                <div class="conf-cell conf-tn">TN: ${c.tn}</div>
                <div class="conf-cell conf-fp">FP: ${c.fp}</div>
                <div class="conf-cell conf-fn">FN: ${c.fn}</div>
                <div class="conf-cell conf-tp">TP: ${c.tp}</div>
            </div>
            <div class="conf-labels"><span>Pred: Normal / Fall</span><span>GT: Normal / Fall</span></div>
        `;
        grid.appendChild(card);
    });
}

function renderModelPickers(data) {
    ['model-picker-cards', 'live-model-picker'].forEach(containerId => {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';
        
        data.models.forEach(model => {
            const card = document.createElement('div');
            card.className = 'mp-card' + (model.id === 'yolo11n-pose' ? ' selected' : '');
            card.dataset.model = model.filename;
            card.innerHTML = `<div class="mp-card-name">${model.name}</div><div class="mp-card-f1">F1: ${model.metrics.f1.toFixed(1)}% • ${model.metrics.fps.toFixed(0)} FPS</div>`;
            
            card.addEventListener('click', () => {
                container.querySelectorAll('.mp-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
            });
            container.appendChild(card);
        });
    });
}

function getSelectedModel(containerId) {
    const sel = document.querySelector(`#${containerId} .mp-card.selected`);
    return sel ? sel.dataset.model : 'models/yolo11n-pose.pt';
}

// ═══════════════════════════════════════════════════════════════════
// 6. FILE UPLOAD & VIDEO ANALYSIS
// ═══════════════════════════════════════════════════════════════════
(function() {
    const zone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const config = document.getElementById('upload-config');
    const analyzeBtn = document.getElementById('analyze-btn');
    const progress = document.getElementById('progress-container');
    const newBtn = document.getElementById('new-analysis-btn');
    let selectedFile = null;

    if (!zone) return;

    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag-over'); if (e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]); });
    zone.addEventListener('click', () => fileInput.click());
    browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
    fileInput.addEventListener('change', () => { if (fileInput.files.length) selectFile(fileInput.files[0]); });
    document.getElementById('file-remove').addEventListener('click', resetUpload);

    function selectFile(file) {
        selectedFile = file;
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-size').textContent = fmtSize(file.size);
        zone.style.display = 'none';
        config.style.display = 'block';
    }

    function resetUpload() {
        selectedFile = null; fileInput.value = '';
        zone.style.display = ''; 
        config.style.display = 'none'; 
        progress.style.display = 'none';
        analyzeBtn.disabled = false;
        document.getElementById('results').style.display = 'none';
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        analyzeBtn.disabled = true;
        config.style.display = 'none';
        progress.style.display = 'block';
        setProg(0, 'Uploading...');
        setStep('upload');

        const fd = new FormData();
        fd.append('video', selectedFile);
        const modelName = getSelectedModel('model-picker-cards').replace('.pt', '');
        fd.append('model', modelName);

        try {
            const result = await uploadXHR(fd);
            setStep('complete'); 
            setProg(100, 'Complete!');
            setTimeout(() => displayResults(result), 500);
        } catch (e) {
            setProg(0, 'Error: ' + e.message);
            analyzeBtn.disabled = false;
            setTimeout(() => { progress.style.display = 'none'; config.style.display = 'block'; }, 3000);
        }
    });

    if (newBtn) {
        newBtn.addEventListener('click', () => {
            resetUpload();
            document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
        });
    }

    function uploadXHR(fd) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.upload.addEventListener('progress', e => { 
                if (e.lengthComputable) setProg(Math.round(e.loaded / e.total * 30), `Uploading ${Math.round(e.loaded / e.total * 100)}%`); 
            });
            xhr.upload.addEventListener('load', () => { 
                setProg(30, 'Processing frames...'); 
                setStep('detect'); 
                startSimProg(); 
            });
            xhr.addEventListener('load', () => { 
                stopSimProg(); 
                xhr.status >= 200 && xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(JSON.parse(xhr.responseText).error || 'Error')); 
            });
            xhr.addEventListener('error', () => { 
                stopSimProg(); 
                reject(new Error('Network error')); 
            });
            xhr.open('POST', '/api/analyze'); 
            xhr.send(fd);
        });
    }

    let simTimer = null;
    function startSimProg() { 
        let p = 30; 
        simTimer = setInterval(() => { 
            if (p < 90) {
                p += Math.random() * 3;
                const v = Math.min(90, Math.round(p));
                setProg(v, `Analyzing frames... ${v}%`);
                if (v > 50) setStep('analyze');
            }
        }, 500); 
    }
    function stopSimProg() { 
        if (simTimer) { clearInterval(simTimer); simTimer = null; } 
    }

    function setProg(pct, detail) {
        const circ = 2 * Math.PI * 54;
        document.getElementById('progress-ring-fill').style.strokeDashoffset = circ - (pct / 100) * circ;
        document.getElementById('progress-percent').textContent = pct + '%';
        document.getElementById('progress-bar').style.width = pct + '%';
        document.getElementById('progress-detail').textContent = detail;
    }

    function setStep(name) {
        const steps = ['upload','detect','analyze','complete'];
        const idx = steps.indexOf(name);
        steps.forEach((s, i) => {
            const el = document.getElementById('step-' + s);
            el.classList.remove('active', 'done');
            if (i < idx) el.classList.add('done');
            else if (i === idx) el.classList.add('active');
        });
    }

    function fmtSize(b) { 
        return b < 1024 ? b + ' B' : b < 1048576 ? (b / 1024).toFixed(1) + ' KB' : (b / 1048576).toFixed(1) + ' MB'; 
    }
})();

// ═══════════════════════════════════════════════════════════════════
// 7. RESULTS DISPLAY & GAUGE / TIMELINE
// ═══════════════════════════════════════════════════════════════════
function displayResults(d) {
    const sec = document.getElementById('results');
    sec.style.display = '';
    setTimeout(() => sec.scrollIntoView({ behavior: 'smooth' }), 100);

    const banner = document.getElementById('verdict-banner');
    banner.className = 'verdict-banner ' + (d.is_fall ? 'fall' : 'no-fall');
    document.getElementById('verdict-icon').textContent = d.is_fall ? '🚨' : '✅';
    document.getElementById('verdict-label').textContent = d.prediction;
    document.getElementById('verdict-detail').textContent = d.is_fall
        ? `Fall at frame ${d.fall_frame} (${d.fall_timestamp}s) • Severity: ${d.severity} • Conf: ${(d.confidence*100).toFixed(1)}%`
        : `No fall across ${d.total_frames} frames • Max conf: ${(d.confidence*100).toFixed(1)}%`;

    drawGauge(d.confidence);
    document.getElementById('metric-confidence').textContent = (d.confidence*100).toFixed(1) + '%';

    const sb = document.getElementById('severity-badge');
    sb.textContent = d.severity; 
    sb.className = 'severity-badge ' + (d.severity ? d.severity.toLowerCase() : 'na');
    
    const bar = document.getElementById('severity-bar');
    bar.style.width = (d.severity_score*100) + '%';
    bar.style.background = d.severity === 'SOFT' ? 'var(--accent-yellow)' : d.severity === 'MODERATE' ? 'var(--accent-orange)' : d.severity === 'SEVERE' ? 'var(--accent-red)' : 'var(--text-muted)';
    document.getElementById('severity-score').textContent = 'Score: ' + (d.severity_score || 0).toFixed(3);

    animCount('metric-nearfalls', d.near_fall_count);
    document.getElementById('metric-immobility').textContent = (d.immobility_seconds || 0).toFixed(1) + 's';
    document.getElementById('immobility-status').textContent = d.immobility_status || 'N/A';
    animCount('metric-fps', Math.round(d.fps_processing));

    document.getElementById('d-res').textContent = d.video_resolution;
    document.getElementById('d-frames').textContent = d.total_frames.toLocaleString();
    document.getElementById('d-fframes').textContent = d.fall_frames.toLocaleString();
    document.getElementById('d-fpct').textContent = d.fall_percentage + '%';
    document.getElementById('d-vel').textContent = (d.peak_velocity || 0).toFixed(3);
    document.getElementById('d-ang').textContent = (d.peak_torso_angle || 0).toFixed(1) + '°';
    document.getElementById('d-pers').textContent = d.persons_tracked;
    document.getElementById('d-model').textContent = (d.model_used || '').replace('.pt','').toUpperCase();
    document.getElementById('d-dev').textContent = (d.device || '').toUpperCase();

    drawTimeline(d.timeline);
    
    const vid = document.getElementById('result-video');
    vid.pause();
    vid.removeAttribute('src');
    vid.load();
    vid.src = '/api/video/' + d.annotated_video;
    vid.type = 'video/mp4';
    vid.load();
    vid.play().catch(() => {}); // auto-start, ignore if blocked
}

function drawGauge(val) {
    const c = document.getElementById('confidence-gauge');
    if (!c) return;
    const ctx = c.getContext('2d');
    const s = 180, cx = s/2, r = 65, lw = 10;
    c.width = s * 2; c.height = s * 2; 
    c.style.width = s + 'px'; c.style.height = s + 'px'; 
    ctx.scale(2, 2);
    const sa = Math.PI * 0.75, ta = Math.PI * 1.5;
    let cur = 0;
    
    (function anim() {
        if (cur < val) { cur += 0.015; if (cur > val) cur = val; }
        ctx.clearRect(0, 0, s, s);
        
        ctx.beginPath(); 
        ctx.arc(cx, cx, r, sa, sa + ta); 
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)'; 
        ctx.lineWidth = lw; 
        ctx.lineCap = 'round'; 
        ctx.stroke();
        
        const g = ctx.createLinearGradient(0, 0, s, s);
        g.addColorStop(0, '#00f2fe'); 
        g.addColorStop(0.5, '#4facfe'); 
        g.addColorStop(1, '#a855f7');
        
        ctx.beginPath(); 
        ctx.arc(cx, cx, r, sa, sa + ta * cur); 
        ctx.strokeStyle = g; 
        ctx.lineWidth = lw; 
        ctx.lineCap = 'round'; 
        ctx.stroke();
        
        ctx.fillStyle = '#f0f0f5'; 
        ctx.font = 'bold 26px "JetBrains Mono", monospace'; 
        ctx.textAlign = 'center'; 
        ctx.textBaseline = 'middle';
        ctx.fillText(Math.round(cur * 100) + '%', cx, cx);
        
        if (cur < val) requestAnimationFrame(anim);
    })();
}

function drawTimeline(tl) {
    const canvas = document.getElementById('timeline-canvas');
    if (!canvas) return;
    const wrap = document.querySelector('.timeline-chart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = wrap.getBoundingClientRect().width - 40, h = 180;
    
    canvas.width = w * dpr; 
    canvas.height = h * dpr; 
    canvas.style.width = w + 'px'; 
    canvas.style.height = h + 'px'; 
    ctx.scale(dpr, dpr);
    
    if (!tl || !tl.length) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)'; 
        ctx.font = '14px Inter'; 
        ctx.textAlign = 'center'; 
        ctx.fillText('No timeline data', w/2, h/2); 
        return;
    }
    
    const pad = {t: 16, r: 16, b: 36, l: 44}, cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
    const sc = {NORMAL: '#00f2fe', DESCENDING: '#fbbf24', IMPACT: '#f97316', LYING: '#f97316', 'CONFIRMED FALL': '#ef4444'};

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)'; 
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + ch / 4 * i; 
        ctx.beginPath(); 
        ctx.moveTo(pad.l, y); 
        ctx.lineTo(pad.l + cw, y); 
        ctx.stroke();
    }
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.25)'; 
    ctx.font = '9px "JetBrains Mono"'; 
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 4; i++) {
        ctx.fillText((1 - i/4) * 100 + '%', pad.l - 6, pad.t + ch / 4 * i + 3);
    }

    ctx.textAlign = 'center';
    const xStep = Math.max(1, Math.floor(tl.length / 8));
    for (let i = 0; i < tl.length; i += xStep) {
        ctx.fillText((tl[i].time || 0).toFixed(1) + 's', pad.l + (i / (tl.length - 1)) * cw, h - 6);
    }

    ctx.beginPath();
    const g = ctx.createLinearGradient(pad.l, 0, pad.l + cw, 0);
    g.addColorStop(0, '#00f2fe'); 
    g.addColorStop(0.5, '#4facfe'); 
    g.addColorStop(1, '#a855f7');
    ctx.strokeStyle = g; 
    ctx.lineWidth = 2;
    
    tl.forEach((p, i) => {
        const x = pad.l + (i / (tl.length - 1)) * cw;
        const y = pad.t + ch - (p.confidence || 0) * ch;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.lineTo(pad.l + cw, pad.t + ch); 
    ctx.lineTo(pad.l, pad.t + ch); 
    ctx.closePath();
    
    const ag = ctx.createLinearGradient(0, pad.t, 0, pad.t + ch);
    ag.addColorStop(0, 'rgba(0, 242, 254, 0.12)'); 
    ag.addColorStop(1, 'rgba(0, 242, 254, 0)');
    ctx.fillStyle = ag; 
    ctx.fill();

    tl.forEach((p, i) => {
        if (p.fall_activity || p.state !== 'NORMAL') {
            const x = pad.l + (i / (tl.length - 1)) * cw;
            const y = pad.t + ch - (p.confidence || 0) * ch;
            ctx.beginPath(); 
            ctx.arc(x, y, 3.5, 0, Math.PI * 2); 
            ctx.fillStyle = sc[p.state] || '#00f2fe'; 
            ctx.fill();
            if (p.state === 'CONFIRMED FALL') {
                ctx.beginPath(); 
                ctx.arc(x, y, 7, 0, Math.PI * 2); 
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.35)'; 
                ctx.lineWidth = 2; 
                ctx.stroke();
            }
        }
    });
}

function animCount(id, target) {
    const el = document.getElementById(id); 
    if (!el) return;
    const start = performance.now();
    (function step(ts) {
        const p = Math.min((ts - start) / 1000, 1);
        el.textContent = Math.round((1 - Math.pow(1 - p, 3)) * target);
        if (p < 1) requestAnimationFrame(step);
    })(start);
}

// ═══════════════════════════════════════════════════════════════════
// 8. LIVE WEBCAM DETECTION
// ═══════════════════════════════════════════════════════════════════
(function() {
    const startBtn = document.getElementById('start-webcam-btn');
    const stopBtn = document.getElementById('stop-webcam-btn');
    const setup = document.getElementById('live-setup');
    const feed = document.getElementById('live-feed');
    const video = document.getElementById('webcam-video');
    const hiddenCanvas = document.getElementById('webcam-canvas-hidden');
    const annotatedImg = document.getElementById('annotated-feed');
    const alertCard = document.getElementById('alert-card');
    const feedState = document.getElementById('feed-state');

    if (!startBtn) return;

    let stream = null, running = false, sessionId = 'sess-' + Date.now();
    let totalFalls = 0, totalNearFalls = 0;

    document.querySelectorAll('.setup-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.setup-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.setup-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
        });
    });

    startBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
            video.srcObject = stream;
            await video.play();

            setup.style.display = 'none';
            feed.style.display = 'block';
            running = true;
            totalFalls = 0; 
            totalNearFalls = 0;
            sessionId = 'sess-' + Date.now();

            const selModel = getSelectedModel('live-model-picker');
            document.getElementById('live-model-name').textContent = selModel.replace('.pt','').toUpperCase();

            fetch('/api/reset-live', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({session_id: sessionId}) });

            processLoop(selModel);
        } catch (e) {
            alert('Camera access denied or unavailable. Please allow camera permissions and try again.\n\nError: ' + e.message);
        }
    });

    stopBtn.addEventListener('click', () => {
        running = false;
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
        feed.style.display = 'none';
        setup.style.display = 'block';
    });

    async function processLoop(modelName) {
        if (!running) return;

        const ctx = hiddenCanvas.getContext('2d');
        hiddenCanvas.width = video.videoWidth || 640;
        hiddenCanvas.height = video.videoHeight || 480;
        ctx.drawImage(video, 0, 0);

        hiddenCanvas.toBlob(async (blob) => {
            if (!blob || !running) { setTimeout(() => processLoop(modelName), 100); return; }

            const fd = new FormData();
            fd.append('frame', blob, 'frame.jpg');
            fd.append('session_id', sessionId);
            fd.append('model', modelName);

            try {
                const resp = await fetch('/api/detect-frame', { method: 'POST', body: fd });
                const data = await resp.json();

                if (data.frame) {
                    annotatedImg.src = 'data:image/jpeg;base64,' + data.frame;
                }

                if (data.detection) {
                    const det = data.detection;
                    document.getElementById('live-persons').textContent = det.persons || 0;

                    let maxConf = 0, mainState = 'NORMAL';
                    let nearFalls = 0;
                    if (det.states) {
                        Object.values(det.states).forEach(s => {
                            if (s.confidence > maxConf) { maxConf = s.confidence; mainState = s.state; }
                            nearFalls += s.near_falls || 0;
                        });
                    }
                    totalNearFalls = nearFalls;

                    document.getElementById('live-confidence').textContent = (maxConf * 100).toFixed(1) + '%';
                    document.getElementById('live-status').textContent = mainState;
                    document.getElementById('live-nearfalls').textContent = totalNearFalls;

                    if (det.has_fall) {
                        totalFalls++;
                        feedState.textContent = '🚨 FALL'; 
                        feedState.className = 'feed-state fall';
                        alertCard.style.display = 'block';
                    } else {
                        feedState.textContent = mainState; 
                        feedState.className = 'feed-state normal';
                        alertCard.style.display = 'none';
                    }
                    document.getElementById('live-falls').textContent = totalFalls;
                }
            } catch (e) {
                console.warn('Frame processing error:', e);
            }

            if (running) setTimeout(() => processLoop(modelName), 80); // ~12 FPS
        }, 'image/jpeg', 0.7);
    }
})();

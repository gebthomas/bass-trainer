"""JavaScript template for the Take Comparator report."""

_COMPARATOR_JS = """\
(function() {
    var audioA = document.getElementById('audio-bass-a');
    var audioB = document.getElementById('audio-bass-b');
    var audioS = document.getElementById('audio-song');
    var audioSB = document.getElementById('audio-song-b');
    var audioStereoA = document.getElementById('audio-stereo-a');
    var audioStereoB = document.getElementById('audio-stereo-b');
    var allAudio = [audioA, audioB, audioS, audioSB, audioStereoA, audioStereoB].filter(function(a){return a;});
    var duration = %DURATION%;
    var plotLeft = %PLOT_LEFT%;
    var plotWidth = %PLOT_WIDTH%;
    var cursors = document.querySelectorAll('.cursor');
    var svgs = document.querySelectorAll('.plot-svg');
    var tooltip = document.getElementById('tooltip');
    var loopRects = document.querySelectorAll('.loop-region');

    function timeToX(t) { return plotLeft + (t / duration) * plotWidth; }
    function xToTime(x) {
        return Math.max(0, Math.min(((x - plotLeft) / plotWidth) * duration, duration));
    }

    /* ── Channel state ───────────────────────────────────────────────── */
    var ch = {bass_a:true, bass_b:true, song:false, song_b:false, stereo_a:false, stereo_b:false};

    function applyChannels() {
        if (audioA) audioA.muted = !ch.bass_a;
        if (audioB) audioB.muted = !ch.bass_b;
        if (audioS) audioS.muted = !ch.song;
        if (audioSB) audioSB.muted = !ch.song_b;
        if (audioStereoA) audioStereoA.muted = !ch.stereo_a;
        if (audioStereoB) audioStereoB.muted = !ch.stereo_b;
        var toggles = document.querySelectorAll('.ch-toggle');
        for (var i = 0; i < toggles.length; i++) {
            var key = toggles[i].dataset.ch;
            toggles[i].classList.toggle('active', !!ch[key]);
        }
        updatePresetHighlight();
    }

    function setChannels(bass_a, bass_b, song, song_b, stereo_a, stereo_b) {
        ch.bass_a=bass_a; ch.bass_b=bass_b; ch.song=song;
        ch.song_b=song_b; ch.stereo_a=stereo_a; ch.stereo_b=stereo_b;
        applyChannels();
    }

    function updatePresetHighlight() {
        var presets = document.querySelectorAll('.preset-btn');
        for (var i = 0; i < presets.length; i++) presets[i].classList.remove('active');
    }

    /* ── Bind channel toggles ────────────────────────────────────────── */
    var toggles = document.querySelectorAll('.ch-toggle');
    for (var ti = 0; ti < toggles.length; ti++) {
        (function(el) {
            el.addEventListener('click', function() {
                var key = el.dataset.ch;
                ch[key] = !ch[key];
                applyChannels();
            });
        })(toggles[ti]);
    }

    /* ── Bind preset buttons ─────────────────────────────────────────── */
    var presetDefs = {
        'preset-stereo-a':  [false,false,false,false,true, false],
        'preset-stereo-b':  [false,false,false,false,false,true],
        'preset-bass-a':    [true, false,false,false,false,false],
        'preset-bass-b':    [false,true, false,false,false,false],
        'preset-ab':        [true, true, false,false,false,false],
        'preset-abs':       [true, true, true, false,false,false],
        'preset-song':      [false,false,true, false,false,false],
        'preset-sync-check':[false,false,true, true, false,false]
    };
    for (var pid in presetDefs) {
        (function(id, args) {
            var el = document.getElementById(id);
            if (el) el.addEventListener('click', function() {
                setChannels.apply(null, args);
                var presets = document.querySelectorAll('.preset-btn');
                for (var i = 0; i < presets.length; i++) presets[i].classList.remove('active');
                el.classList.add('active');
            });
        })(pid, presetDefs[pid]);
    }

    /* ── Audio sync (all pre-aligned, no offset needed) ──────────────── */
    function syncAll() {
        if (!audioA) return;
        var t = audioA.currentTime;
        for (var i = 0; i < allAudio.length; i++) {
            if (allAudio[i] !== audioA) allAudio[i].currentTime = t;
        }
    }

    function playActive() {
        if (!audioA) return;
        syncAll();
        applyChannels();
        for (var i = 0; i < allAudio.length; i++) allAudio[i].play();
        requestAnimationFrame(updateCursor);
    }

    function pauseAll() {
        for (var i = 0; i < allAudio.length; i++) allAudio[i].pause();
        updateCursor();
    }

    var playBtn = document.getElementById('play-btn');
    if (playBtn) playBtn.addEventListener('click', function() {
        if (audioA && audioA.paused) playActive(); else pauseAll();
    });

    /* ── Cursor sync ─────────────────────────────────────────────────── */
    function updateCursor() {
        if (!audioA) return;
        var x = timeToX(audioA.currentTime);
        for (var i = 0; i < cursors.length; i++) {
            cursors[i].setAttribute('x1', x);
            cursors[i].setAttribute('x2', x);
            cursors[i].setAttribute('opacity', '0.8');
        }
        var anyPlaying = false;
        for (var j = 0; j < allAudio.length; j++) {
            if (!allAudio[j].paused) { anyPlaying = true; break; }
        }
        if (anyPlaying) requestAnimationFrame(updateCursor);
    }
    if (audioA) {
        audioA.addEventListener('play', function() { requestAnimationFrame(updateCursor); });
        audioA.addEventListener('seeked', function() { syncAll(); updateCursor(); });
        audioA.addEventListener('pause', function() { pauseAll(); });
        audioA.addEventListener('timeupdate', function() {
            updateCursor();
            if (loopEnabled && audioA.currentTime >= loopEnd) {
                audioA.currentTime = loopStart;
                syncAll();
            }
        });
    }

    /* ── Click-to-seek ───────────────────────────────────────────────── */
    for (var s = 0; s < svgs.length; s++) {
        (function(svg) {
            svg.addEventListener('click', function(e) {
                if (!audioA) return;
                var pt = svg.createSVGPoint();
                pt.x = e.clientX; pt.y = e.clientY;
                var svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());
                audioA.currentTime = xToTime(svgPt.x);
                syncAll();
                updateCursor();
            });
        })(svgs[s]);
    }

    /* ── Speed controls ──────────────────────────────────────────────── */
    var speedLabel = document.getElementById('speed-label');
    var speedBtns = document.querySelectorAll('.speed-btn');
    for (var sb = 0; sb < speedBtns.length; sb++) {
        (function(btn) {
            btn.addEventListener('click', function() {
                var rate = parseFloat(btn.dataset.speed);
                for (var i = 0; i < allAudio.length; i++) allAudio[i].playbackRate = rate;
                if (speedLabel) speedLabel.textContent = btn.dataset.speed + 'x';
                for (var j = 0; j < speedBtns.length; j++)
                    speedBtns[j].classList.remove('active');
                btn.classList.add('active');
            });
        })(speedBtns[sb]);
    }

    /* ── Loop controls ───────────────────────────────────────────────── */
    var loopEnabled = false;
    var loopStart = 0;
    var loopEnd = duration;
    var loopBtn = document.getElementById('loop-toggle');
    var loopStartIn = document.getElementById('loop-start-in');
    var loopEndIn = document.getElementById('loop-end-in');

    function updateLoopDisplay() {
        if (loopBtn) loopBtn.classList.toggle('active', loopEnabled);
        if (loopStartIn) loopStartIn.value = loopStart.toFixed(2);
        if (loopEndIn) loopEndIn.value = loopEnd.toFixed(2);
        for (var lr = 0; lr < loopRects.length; lr++) {
            var r = loopRects[lr];
            if (loopEnabled) {
                r.setAttribute('x', timeToX(loopStart));
                r.setAttribute('width', timeToX(loopEnd) - timeToX(loopStart));
                r.setAttribute('opacity', '0.08');
            } else {
                r.setAttribute('opacity', '0');
            }
        }
    }
    function setLoopStart(t) { loopStart = Math.max(0, Math.min(t, loopEnd - 0.05)); updateLoopDisplay(); }
    function setLoopEnd(t) { loopEnd = Math.max(loopStart + 0.05, Math.min(t, duration)); updateLoopDisplay(); }
    function setLoopRegion(start, end) { loopStart = start; loopEnd = end; loopEnabled = true; updateLoopDisplay(); }

    if (loopBtn) loopBtn.addEventListener('click', function() {
        loopEnabled = !loopEnabled; updateLoopDisplay();
    });
    var lsBtn = document.getElementById('loop-set-start');
    var leBtn = document.getElementById('loop-set-end');
    var lrBtn = document.getElementById('loop-reset');
    var lrestartBtn = document.getElementById('loop-restart');
    if (lsBtn) lsBtn.addEventListener('click', function() { if(audioA) setLoopStart(audioA.currentTime); });
    if (leBtn) leBtn.addEventListener('click', function() { if(audioA) setLoopEnd(audioA.currentTime); });
    if (lrBtn) lrBtn.addEventListener('click', function() { loopStart=0; loopEnd=duration; loopEnabled=false; updateLoopDisplay(); });
    if (lrestartBtn) lrestartBtn.addEventListener('click', function() {
        if(audioA){ audioA.currentTime = loopEnabled ? loopStart : 0; syncAll(); updateCursor(); }
    });
    if (loopStartIn) loopStartIn.addEventListener('change', function() { setLoopStart(parseFloat(this.value)||0); });
    if (loopEndIn) loopEndIn.addEventListener('change', function() { setLoopEnd(parseFloat(this.value)||duration); });
    updateLoopDisplay();

    /* ── Disagreement cards ──────────────────────────────────────────── */
    var cards = document.querySelectorAll('.disagreement-card');
    var selectedCard = null;
    var ZOOM_RADIUS = 2.0;

    function selectCard(card) {
        if (selectedCard) selectedCard.classList.remove('selected');
        card.classList.add('selected');
        selectedCard = card;
    }

    for (var ci = 0; ci < cards.length; ci++) {
        (function(card) {
            card.addEventListener('click', function() {
                selectCard(card);
                var t = parseFloat(card.dataset.time);
                var leadIn = Math.max(0, t - 0.5);
                if (audioA) {
                    audioA.currentTime = leadIn;
                    syncAll();
                    updateCursor();
                    playActive();
                }
            });
        })(cards[ci]);
    }

    /* ── Zoom-to-event loop ──────────────────────────────────────────── */
    var zoomBtn = document.getElementById('zoom-event-btn');
    if (zoomBtn) zoomBtn.addEventListener('click', function() {
        if (!selectedCard) return;
        var t = parseFloat(selectedCard.dataset.time);
        var lo = Math.max(0, t - ZOOM_RADIUS);
        var hi = Math.min(duration, t + ZOOM_RADIUS);
        setLoopRegion(lo, hi);
        if (audioA) {
            audioA.currentTime = lo;
            syncAll();
            updateCursor();
            playActive();
        }
    });

    /* ── Display filter modes ────────────────────────────────────────── */
    var filterBtns = document.querySelectorAll('.filter-btn');
    var activeFilters = {matched:true, a_only:true, b_only:true, ambiguous:true, noise:false};

    function applyFilters() {
        for (var c = 0; c < cards.length; c++) {
            var cc = cards[c].dataset.category;
            cards[c].style.display = activeFilters[cc] ? '' : 'none';
        }
        for (var fb = 0; fb < filterBtns.length; fb++) {
            var cat = filterBtns[fb].dataset.category;
            filterBtns[fb].classList.toggle('active', activeFilters[cat]);
        }
    }

    for (var fb = 0; fb < filterBtns.length; fb++) {
        (function(btn) {
            btn.addEventListener('click', function() {
                var cat = btn.dataset.category;
                activeFilters[cat] = !activeFilters[cat];
                applyFilters();
            });
        })(filterBtns[fb]);
    }

    /* ── Quick-filter preset buttons ─────────────────────────────────── */
    var qfDefs = {
        'qf-all':           {matched:true, a_only:true, b_only:true, ambiguous:true, noise:true},
        'qf-disagreements':  {matched:false, a_only:true, b_only:true, ambiguous:true, noise:false},
        'qf-matched':       {matched:true, a_only:false, b_only:false, ambiguous:false, noise:false}
    };
    for (var qfId in qfDefs) {
        (function(id, filters) {
            var el = document.getElementById(id);
            if (el) el.addEventListener('click', function() {
                for (var k in filters) activeFilters[k] = filters[k];
                applyFilters();
            });
        })(qfId, qfDefs[qfId]);
    }

    applyFilters();

    /* ── Tooltip ──────────────────────────────────────────────────────── */
    var markers = document.querySelectorAll('.onset-marker');
    for (var m = 0; m < markers.length; m++) {
        (function(el) {
            el.addEventListener('mouseenter', function(e) {
                var d = el.dataset;
                tooltip.innerHTML =
                    '<b>' + d.take + ' #' + d.id + ' at ' + d.time + 's</b><br>' +
                    'Strength: ' + d.strength + '<br>' +
                    'Amplitude: ' + d.amp + ' dB';
                tooltip.style.display = 'block';
                moveTooltip(e);
            });
            el.addEventListener('mousemove', moveTooltip);
            el.addEventListener('mouseleave', function() {
                tooltip.style.display = 'none';
            });
        })(markers[m]);
    }
    function moveTooltip(e) {
        var x = e.clientX + 14, y = e.clientY - 14;
        if (x + 270 > window.innerWidth) x = e.clientX - 270;
        if (y < 0) y = e.clientY + 20;
        tooltip.style.left = x + 'px'; tooltip.style.top = y + 'px';
    }

    /* ── Hotkeys ──────────────────────────────────────────────────────── */
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        var k = e.key;
        if (k === ' ') {
            e.preventDefault();
            if (audioA && audioA.paused) playActive(); else pauseAll();
        }
        else if (k === 'r' || k === 'R') {
            e.preventDefault();
            if (audioA) { audioA.currentTime = loopEnabled ? loopStart : 0; syncAll(); updateCursor(); }
        }
        else if (k === 'l' || k === 'L') {
            e.preventDefault(); loopEnabled = !loopEnabled; updateLoopDisplay();
        }
        else if (k === '[') { e.preventDefault(); if(audioA) setLoopStart(audioA.currentTime); }
        else if (k === ']') { e.preventDefault(); if(audioA) setLoopEnd(audioA.currentTime); }
        else if (k === '1') { e.preventDefault(); ch.bass_a = !ch.bass_a; applyChannels(); }
        else if (k === '2') { e.preventDefault(); ch.bass_b = !ch.bass_b; applyChannels(); }
        else if (k === '3') { e.preventDefault(); ch.song = !ch.song; applyChannels(); }
        else if (k === 'z' || k === 'Z') {
            e.preventDefault();
            var zb = document.getElementById('zoom-event-btn');
            if (zb) zb.click();
        }
        else if (k === 'ArrowDown') {
            e.preventDefault();
            if (selectedCard) {
                var next = selectedCard.nextElementSibling;
                while (next && next.style.display === 'none') next = next.nextElementSibling;
                if (next && next.classList.contains('disagreement-card')) {
                    selectCard(next);
                    next.scrollIntoView({block:'nearest'});
                }
            } else if (cards.length > 0) { selectCard(cards[0]); }
        }
        else if (k === 'ArrowUp') {
            e.preventDefault();
            if (selectedCard) {
                var prev = selectedCard.previousElementSibling;
                while (prev && prev.style.display === 'none') prev = prev.previousElementSibling;
                if (prev && prev.classList.contains('disagreement-card')) {
                    selectCard(prev);
                    prev.scrollIntoView({block:'nearest'});
                }
            }
        }
        else if (k === 'ArrowLeft' && e.shiftKey) {
            e.preventDefault();
            if(audioA){audioA.currentTime=Math.max(0,audioA.currentTime-0.25);syncAll();updateCursor();}
        }
        else if (k === 'ArrowRight' && e.shiftKey) {
            e.preventDefault();
            if(audioA){audioA.currentTime=Math.min(duration,audioA.currentTime+0.25);syncAll();updateCursor();}
        }
    });

    applyChannels();
})();
"""

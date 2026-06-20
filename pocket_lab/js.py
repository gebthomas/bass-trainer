"""JavaScript template for the Beat Microscope inspector report."""

_JS_TEMPLATE = """\
(function() {
    var audio = document.getElementById('audio-player');
    var duration = %DURATION%;
    var plotLeft = %PLOT_LEFT%;
    var plotWidth = %PLOT_WIDTH%;
    var STORAGE_KEY = '%STORAGE_KEY%';
    var SOURCE_FILE = '%SOURCE_FILE%';
    var BPM = %BPM%;
    var BPM_INT = %BEATS_PER_MEASURE%;
    var SHUF = %SHUFFLE_FRACTION%;
    var beatZero = %BEAT_ZERO%;
    var winStart = %WINDOW_START%;
    var beatS = 60.0 / BPM;
    var cursors = document.querySelectorAll('.cursor');
    var svgs = document.querySelectorAll('.plot-svg');
    var tooltip = document.getElementById('tooltip');
    var loopRects = document.querySelectorAll('.loop-region');

    function timeToX(t) { return plotLeft + (t / duration) * plotWidth; }
    function xToTime(x) {
        return Math.max(0, Math.min(((x - plotLeft) / plotWidth) * duration, duration));
    }

    /* ── Onset id list (sorted) for arrow-key nav ────────────────────── */
    var onsetIds = [];
    var seen = {};
    var allMarkers = document.querySelectorAll('.onset-marker');
    for (var oi = 0; oi < allMarkers.length; oi++) {
        var oid = allMarkers[oi].dataset.id;
        if (!seen[oid]) { onsetIds.push(parseInt(oid)); seen[oid] = 1; }
    }
    onsetIds.sort(function(a,b){return a-b;});
    var markers = allMarkers;

    /* ── Cursor sync ─────────────────────────────────────────────────── */
    function updateCursor() {
        if (!audio) return;
        var x = timeToX(audio.currentTime);
        for (var i = 0; i < cursors.length; i++) {
            cursors[i].setAttribute('x1', x);
            cursors[i].setAttribute('x2', x);
            cursors[i].setAttribute('opacity', '0.8');
        }
        if (!audio.paused) requestAnimationFrame(updateCursor);
    }
    if (audio) {
        audio.addEventListener('play', function() { requestAnimationFrame(updateCursor); });
        audio.addEventListener('seeked', updateCursor);
        audio.addEventListener('pause', updateCursor);
        audio.addEventListener('timeupdate', function() {
            updateCursor();
            if (loopEnabled && audio.currentTime >= loopEnd)
                audio.currentTime = loopStart;
        });
    }

    /* ── Click-to-seek ───────────────────────────────────────────────── */
    for (var s = 0; s < svgs.length; s++) {
        (function(svg) {
            svg.addEventListener('click', function(e) {
                if (!audio) return;
                var pt = svg.createSVGPoint();
                pt.x = e.clientX; pt.y = e.clientY;
                var svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());
                audio.currentTime = xToTime(svgPt.x);
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
                if (!audio) return;
                audio.playbackRate = parseFloat(btn.dataset.speed);
                if (speedLabel) speedLabel.textContent = btn.dataset.speed + 'x';
                for (var j = 0; j < speedBtns.length; j++)
                    speedBtns[j].classList.remove('active');
                btn.classList.add('active');
            });
        })(speedBtns[sb]);
    }

    /* ── Channel switching ───────────────────────────────────────────── */
    var channelSel = document.getElementById('channel-select');
    if (channelSel && audio) {
        channelSel.addEventListener('change', function() {
            var cur = audio.currentTime;
            var playing = !audio.paused;
            audio.src = channelSel.value;
            audio.currentTime = cur;
            if (playing) audio.play();
        });
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

    if (loopBtn) loopBtn.addEventListener('click', function() {
        loopEnabled = !loopEnabled; updateLoopDisplay();
    });
    var lsBtn = document.getElementById('loop-set-start');
    var leBtn = document.getElementById('loop-set-end');
    var lrBtn = document.getElementById('loop-reset');
    var lrestartBtn = document.getElementById('loop-restart');
    if (lsBtn) lsBtn.addEventListener('click', function() { if(audio) setLoopStart(audio.currentTime); });
    if (leBtn) leBtn.addEventListener('click', function() { if(audio) setLoopEnd(audio.currentTime); });
    if (lrBtn) lrBtn.addEventListener('click', function() { loopStart=0; loopEnd=duration; updateLoopDisplay(); });
    if (lrestartBtn) lrestartBtn.addEventListener('click', function() {
        if(audio){ audio.currentTime = loopEnabled ? loopStart : 0; updateCursor(); }
    });
    if (loopStartIn) loopStartIn.addEventListener('change', function() { setLoopStart(parseFloat(this.value)||0); });
    if (loopEndIn) loopEndIn.addEventListener('change', function() { setLoopEnd(parseFloat(this.value)||duration); });
    updateLoopDisplay();

    /* ── Tooltip ──────────────────────────────────────────────────────── */
    for (var m = 0; m < markers.length; m++) {
        (function(el) {
            el.addEventListener('mouseenter', function(e) {
                var d = el.dataset;
                var ann = annotations[d.id];
                var extra = ann ? '<br>Annotation: <b>' + ann.label + '</b>' : '';
                tooltip.innerHTML =
                    '<b>Onset #' + d.id + ' at ' + d.time + 's</b><br>' +
                    'Measure ' + d.measure + ', Beat ' + d.beat + '<br>' +
                    'Offset: ' + d.offset + ' ms<br>' +
                    'Beat fraction: ' + d.fraction + '<br>' +
                    'Label: <b>' + d.label + '</b><br>' +
                    'Strength: ' + d.strength + extra;
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

    /* ── Annotations ─────────────────────────────────────────────────── */
    var LABEL_COLORS = {
        'true_attack':'#2ecc71', 'string_noise':'#f1c40f',
        'passing_note':'#9b59b6', 'downbeat':'#ff9f1c',
        'beat3':'#4895ef', 'fill':'#4cc9f0',
        'ignore':'#555', 'uncertain':'#aaa'
    };
    var LABEL_KEYS = ['true_attack','string_noise','passing_note','downbeat',
                      'beat3','fill','ignore','uncertain'];
    var annotations = {};
    var selectedOnsetId = null;
    var panel = document.getElementById('annotation-panel');
    var annInfo = document.getElementById('ann-info');
    var annNote = document.getElementById('ann-note');
    var annStatus = document.getElementById('ann-status');
    var selStatus = document.getElementById('selected-status');

    function applyAnnotationColors() {
        for (var k = 0; k < markers.length; k++) {
            var el = markers[k], id = el.dataset.id;
            if (annotations[id]) {
                el.setAttribute('fill', LABEL_COLORS[annotations[id].label]||'#ff6b6b');
                el.setAttribute('stroke', '#fff');
                el.setAttribute('stroke-width', '1.5');
            } else {
                el.setAttribute('fill', '#ff6b6b');
                el.setAttribute('stroke', '#fff');
                el.setAttribute('stroke-width', '0.5');
            }
        }
    }

    function updateSelStatus() {
        if (!selStatus) return;
        if (selectedOnsetId === null) { selStatus.textContent = 'No onset selected'; return; }
        var el = document.querySelector('.onset-marker[data-id="'+selectedOnsetId+'"]');
        if (!el) { selStatus.textContent = 'No onset selected'; return; }
        var d = el.dataset, ann = annotations[selectedOnsetId];
        selStatus.innerHTML = 'Selected: <b>#'+selectedOnsetId+'</b> t='+d.time+
            's M'+d.measure+' B'+d.beat+
            (ann ? ' — <b>'+ann.label+'</b>' : ' — unannotated')+
            (ann && ann.note ? ' "'+ann.note+'"' : '');
    }

    function selectOnset(id) {
        var prev = document.querySelectorAll('.onset-marker[data-id="'+selectedOnsetId+'"]');
        for (var p = 0; p < prev.length; p++)
            prev[p].setAttribute('stroke-width', annotations[selectedOnsetId] ? '1.5' : '0.5');
        selectedOnsetId = id;
        var els = document.querySelectorAll('.onset-marker[data-id="'+id+'"]');
        for (var q = 0; q < els.length; q++) els[q].setAttribute('stroke-width', '3');
        var el = els[0]; if (!el) return;
        var d = el.dataset;
        if (annInfo) annInfo.innerHTML = '<b>Onset #'+id+'</b> at '+d.time+
            's — M'+d.measure+' B'+d.beat+' ('+d.label+')';
        if (annNote) annNote.value = annotations[id] ? (annotations[id].note||'') : '';
        var btns = document.querySelectorAll('.ann-btn');
        for (var b = 0; b < btns.length; b++) {
            btns[b].classList.remove('active');
            if (annotations[id] && btns[b].dataset.label === annotations[id].label)
                btns[b].classList.add('active');
        }
        if (panel) panel.style.display = 'block';
        updateSelStatus();
    }

    function assignLabel(label) {
        if (selectedOnsetId === null) return;
        var el = document.querySelector('.onset-marker[data-id="'+selectedOnsetId+'"]');
        if (!el) return;
        annotations[selectedOnsetId] = {
            time_s: parseFloat(el.dataset.time),
            detected_onset_id: parseInt(selectedOnsetId),
            label: label, note: annNote ? annNote.value : '',
            created_by: 'user', created_at: new Date().toISOString()
        };
        var btns = document.querySelectorAll('.ann-btn');
        for (var j = 0; j < btns.length; j++) {
            btns[j].classList.remove('active');
            if (btns[j].dataset.label === label) btns[j].classList.add('active');
        }
        applyAnnotationColors(); saveToStorage(); updateSelStatus();
        if (annStatus) annStatus.textContent = 'Saved: #'+selectedOnsetId+' \\u2192 '+label;
    }

    function clearAnnotation() {
        if (selectedOnsetId === null) return;
        delete annotations[selectedOnsetId];
        var btns = document.querySelectorAll('.ann-btn');
        for (var j = 0; j < btns.length; j++) btns[j].classList.remove('active');
        applyAnnotationColors(); saveToStorage(); updateSelStatus();
        if (annStatus) annStatus.textContent = 'Cleared #'+selectedOnsetId;
    }

    for (var mc = 0; mc < markers.length; mc++) {
        (function(el) {
            el.addEventListener('click', function(e) {
                e.stopPropagation(); selectOnset(el.dataset.id);
            });
        })(markers[mc]);
    }

    var annBtns = document.querySelectorAll('.ann-btn');
    for (var ab = 0; ab < annBtns.length; ab++) {
        (function(btn) {
            btn.addEventListener('click', function() { assignLabel(btn.dataset.label); });
        })(annBtns[ab]);
    }

    if (annNote) annNote.addEventListener('change', function() {
        if (selectedOnsetId !== null && annotations[selectedOnsetId]) {
            annotations[selectedOnsetId].note = annNote.value;
            saveToStorage(); updateSelStatus();
        }
    });

    var cancelBtn = document.getElementById('ann-cancel');
    if (cancelBtn) cancelBtn.addEventListener('click', function() {
        if (panel) panel.style.display = 'none';
        var prev = document.querySelectorAll('.onset-marker[data-id="'+selectedOnsetId+'"]');
        for (var p = 0; p < prev.length; p++)
            prev[p].setAttribute('stroke-width', annotations[selectedOnsetId] ? '1.5' : '0.5');
        selectedOnsetId = null; updateSelStatus();
    });

    function saveToStorage() {
        var data = {version:1, source_file:SOURCE_FILE,
            annotations: Object.keys(annotations).map(function(k){ return annotations[k]; })};
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); } catch(e) {}
    }

    try {
        var stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            var parsed = JSON.parse(stored);
            if (parsed.annotations) {
                for (var li = 0; li < parsed.annotations.length; li++) {
                    var a = parsed.annotations[li];
                    annotations[a.detected_onset_id] = a;
                }
                applyAnnotationColors();
            }
        }
    } catch(e) {}

    var exportBtn = document.getElementById('ann-export');
    if (exportBtn) exportBtn.addEventListener('click', function() {
        var data = {version:1, source_file:SOURCE_FILE,
            annotations: Object.keys(annotations).map(function(k){ return annotations[k]; })};
        var blob = new Blob([JSON.stringify(data,null,2)], {type:'application/json'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a'); a.href = url;
        a.download = STORAGE_KEY+'.json';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
        if (annStatus) annStatus.textContent = 'Exported '+Object.keys(annotations).length+' annotations.';
    });

    var importInput = document.getElementById('ann-import');
    if (importInput) importInput.addEventListener('change', function(e) {
        var file = e.target.files[0]; if (!file) return;
        var reader = new FileReader();
        reader.onload = function(ev) {
            try {
                var data = JSON.parse(ev.target.result);
                if (data.annotations) {
                    annotations = {};
                    for (var i = 0; i < data.annotations.length; i++)
                        annotations[data.annotations[i].detected_onset_id] = data.annotations[i];
                    applyAnnotationColors(); saveToStorage();
                    if (annStatus) annStatus.textContent = 'Imported '+data.annotations.length+' annotations.';
                }
            } catch(err) { if (annStatus) annStatus.textContent = 'Import error: '+err.message; }
        };
        reader.readAsText(file);
    });

    /* ── Hotkeys ──────────────────────────────────────────────────────── */
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' ||
            e.target.tagName === 'SELECT') return;
        var k = e.key;
        if (k === ' ') { e.preventDefault(); if(audio){audio.paused?audio.play():audio.pause();} }
        else if (k === 'r' || k === 'R') {
            e.preventDefault();
            if(audio){audio.currentTime=loopEnabled?loopStart:0; updateCursor();}
        }
        else if (k === 'l' || k === 'L') {
            e.preventDefault(); loopEnabled=!loopEnabled; updateLoopDisplay();
        }
        else if (k === '[') { e.preventDefault(); if(audio) setLoopStart(audio.currentTime); }
        else if (k === ']') { e.preventDefault(); if(audio) setLoopEnd(audio.currentTime); }
        else if (k === 'ArrowLeft' && !e.shiftKey) {
            e.preventDefault();
            if (selectedOnsetId !== null) {
                var ci = onsetIds.indexOf(parseInt(selectedOnsetId));
                if (ci > 0) selectOnset(String(onsetIds[ci-1]));
            }
        }
        else if (k === 'ArrowRight' && !e.shiftKey) {
            e.preventDefault();
            if (selectedOnsetId !== null) {
                var ci2 = onsetIds.indexOf(parseInt(selectedOnsetId));
                if (ci2 < onsetIds.length-1) selectOnset(String(onsetIds[ci2+1]));
            }
        }
        else if (k === 'ArrowLeft' && e.shiftKey) {
            e.preventDefault(); if(audio){audio.currentTime=Math.max(0,audio.currentTime-0.25);updateCursor();}
        }
        else if (k === 'ArrowRight' && e.shiftKey) {
            e.preventDefault(); if(audio){audio.currentTime=Math.min(duration,audio.currentTime+0.25);updateCursor();}
        }
        else if (k === 'Delete' || k === 'Backspace') { e.preventDefault(); clearAnnotation(); }
        else if (k >= '1' && k <= '8') {
            e.preventDefault(); assignLabel(LABEL_KEYS[parseInt(k)-1]);
        }
    });

    /* ── Grid calibration ─────────────────────────────────────────────── */
    var calMenu = document.getElementById('grid-calibration-menu');
    var anchorDisp = document.getElementById('grid-anchor-display');
    var calClickTime = 0;
    var gridLines = document.querySelectorAll('.grid-line');
    var gridLabels = document.querySelectorAll('.grid-label');

    function updateAnchorDisplay() {
        if (!anchorDisp) return;
        var rel = -beatZero;
        if (rel < 0) rel += Math.ceil(-rel / (BPM_INT * beatS)) * BPM_INT * beatS;
        var beatIdx = Math.round(rel / beatS);
        var m = Math.floor(beatIdx / BPM_INT) + 1;
        var b = (beatIdx % BPM_INT) + 1;
        anchorDisp.innerHTML = 'Grid anchor: <b>Measure ' + m + ' Beat ' + b +
            '</b> at <b>' + beatZero.toFixed(4) + 's</b>';
        var bzIn = document.getElementById('beat-zero-input');
        if (bzIn) bzIn.value = beatZero.toFixed(4);
    }

    function redrawGrid() {
        for (var g = 0; g < gridLines.length; g++) {
            var gl = gridLines[g];
            var t = parseFloat(gl.dataset.gridTime);
            var kind = gl.dataset.gridKind;
            var beatInMeasure = parseInt(gl.dataset.gridBeat);
            var newRel = t - beatZero;
            var measS = BPM_INT * beatS;
            var gridBeatIdx = Math.round(newRel / beatS);
            var newBeat = ((gridBeatIdx % BPM_INT) + BPM_INT) % BPM_INT;
            var newMeasure = Math.floor(gridBeatIdx / BPM_INT) + 1;
            var newT;
            if (kind === 'subdivision') {
                var parentBeatT = beatZero + gridBeatIdx * beatS;
                newT = parentBeatT + SHUF * beatS;
            } else {
                newT = beatZero + gridBeatIdx * beatS;
            }
            var x = plotLeft + ((newT - winStart) / duration) * plotWidth;
            gl.setAttribute('x1', x);
            gl.setAttribute('x2', x);
            if (x < plotLeft || x > plotLeft + plotWidth) {
                gl.setAttribute('opacity', '0');
            } else {
                gl.setAttribute('opacity', '0.8');
            }
        }
        for (var lb = 0; lb < gridLabels.length; lb++) {
            var lbl = gridLabels[lb];
            var lt = parseFloat(lbl.dataset.gridTime);
            var lRel = lt - beatZero;
            var lBeatIdx = Math.round(lRel / beatS);
            var lMeasure = Math.floor(lBeatIdx / BPM_INT) + 1;
            var lNewT = beatZero + lBeatIdx * beatS;
            var lx = plotLeft + ((lNewT - winStart) / duration) * plotWidth;
            lbl.setAttribute('x', lx);
            lbl.textContent = 'M' + lMeasure;
            if (lx < plotLeft || lx > plotLeft + plotWidth) {
                lbl.setAttribute('opacity', '0');
            } else {
                lbl.setAttribute('opacity', '1');
            }
        }
        updateAnchorDisplay();
    }

    function setGridAnchor(time, beatNumber) {
        beatZero = time - (beatNumber - 1) * beatS;
        redrawGrid();
    }

    function showCalMenu(x, y, time) {
        if (!calMenu) return;
        calClickTime = time;
        calMenu.style.left = x + 'px';
        calMenu.style.top = y + 'px';
        calMenu.style.display = 'block';
    }

    if (calMenu) {
        var items = calMenu.querySelectorAll('.cal-item');
        for (var ci = 0; ci < items.length; ci++) {
            (function(item) {
                item.addEventListener('click', function() {
                    var bn = parseInt(item.dataset.beat);
                    setGridAnchor(calClickTime, bn);
                    calMenu.style.display = 'none';
                });
            })(items[ci]);
        }
        document.addEventListener('click', function() {
            calMenu.style.display = 'none';
        });
    }

    for (var sv = 0; sv < svgs.length; sv++) {
        (function(svg) {
            svg.addEventListener('contextmenu', function(e) {
                e.preventDefault();
                var pt = svg.createSVGPoint();
                pt.x = e.clientX; pt.y = e.clientY;
                var svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());
                var t = xToTime(svgPt.x) + winStart;
                showCalMenu(e.clientX, e.clientY, t);
            });
        })(svgs[sv]);
    }

    var calFromOnsetBtn = document.getElementById('cal-from-selected');
    if (calFromOnsetBtn) {
        calFromOnsetBtn.addEventListener('click', function() {
            if (selectedOnsetId === null) return;
            var el = document.querySelector('.onset-marker[data-id="'+selectedOnsetId+'"]');
            if (!el) return;
            var t = parseFloat(el.dataset.time) + winStart;
            showCalMenu(
                calFromOnsetBtn.getBoundingClientRect().left,
                calFromOnsetBtn.getBoundingClientRect().bottom + 2, t);
        });
    }

    updateAnchorDisplay();
    updateSelStatus();
})();
"""

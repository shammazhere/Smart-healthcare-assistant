/**
 * Baymax Assistant — Voice + Chat Clinical System
 * Talks to the AI gateway (server.py) over WebSocket /ws/web:
 *   mic audio (16 kHz PCM) or typed text -> SQLite RAG records + Gemini -> text reply + spoken audio
 * Progressive Web App (PWA) enabled with offline shell caching & install prompts.
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const el = {
    pill: $('wsPill'),
    pillText: $('wsStatusText'),
    select: $('patientSelect'),
    engineInfo: $('engineInfo'),
    statePill: $('audioStatePill'),
    stateText: $('audioStateText'),
    chat: $('chatScroll'),
    suggestions: $('suggestions'),
    mic: $('micBtn'),
    micHint: $('micHint'),
    soundbars: $('audioSoundbars'),
    input: $('textInput'),
    send: $('sendBtn'),
    patient: $('patientCard'),
    sources: $('sourcesList'),
    convBtn: $('convBtn'),
    newChat: $('newChatBtn'),
    installBtn: $('pwaInstallBtn'),
  };

  const TARGET_RATE = 16000;
  const NO_SPEECH_MSG = "I didn't hear anything. Tap the microphone and try again.";
  const STATE_LABELS = {
    idle: 'Ready',
    listening: 'Listening…',
    transcribing: 'Processing speech…',
    thinking: 'Reviewing records…',
    speaking: 'Speaking…',
  };
  const SECTION_LABELS = {
    profile: 'Profile',
    allergies: 'Allergy Record',
    conditions: 'Diagnosis',
    surgeries: 'Surgical History',
    medications: 'Medication',
    lab_results: 'Lab Vitals',
    vitals_baseline: 'Usual Baseline',
    visits: 'Clinical Visit',
    appointments: 'Appointment',
    care_instructions: 'Care Protocol',
    doctor_notes: 'Physician Note',
  };

  const state = {
    ws: null,
    connected: false,
    patientId: null,
    ui: 'idle',
    recording: false,
    starting: false,
    convMode: false,
    silent: 0,
  };

  // ── PWA Service Worker & Install Prompt ────────────────────────────────────
  let deferredInstallPrompt = null;
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/sw.js').then((reg) => {
        console.log('[PWA] Service Worker registered with scope:', reg.scope);
      }).catch((err) => {
        console.warn('[PWA] Service Worker registration failed:', err);
      });
    });
  }

  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredInstallPrompt = e;
    if (el.installBtn) {
      el.installBtn.style.display = 'inline-flex';
    }
  });

  if (el.installBtn) {
    el.installBtn.addEventListener('click', async () => {
      if (!deferredInstallPrompt) return;
      deferredInstallPrompt.prompt();
      const choice = await deferredInstallPrompt.userChoice;
      if (choice.outcome === 'accepted') {
        el.installBtn.style.display = 'none';
      }
      deferredInstallPrompt = null;
    });
  }

  // ── Tiny DOM helper ────────────────────────────────────────────────────────
  function h(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  // ── UI state ───────────────────────────────────────────────────────────────
  function setState(s) {
    state.ui = s;
    if (el.stateText) el.stateText.textContent = STATE_LABELS[s] || s;
    const cls = s === 'transcribing' ? 'thinking' : s;
    if (el.statePill) el.statePill.className = 'audio-state-pill' + (cls !== 'idle' ? ' ' + cls : '');
    if (el.mic) el.mic.classList.toggle('active', state.recording);
    if (el.soundbars) el.soundbars.classList.toggle('active', state.recording);
  }

  function setConnection(kind, label) {
    if (el.pill) el.pill.className = 'status-pill ' + kind;
    if (el.pillText) el.pillText.textContent = label;
  }

  function scrollChat() {
    if (el.chat) el.chat.scrollTop = el.chat.scrollHeight;
  }

  function addMessage(kind, text, pending) {
    const msg = h('div', 'dialogue-msg ' + kind + (pending ? ' pending' : ''));
    const who = kind === 'user' ? 'Patient' : kind === 'robot' ? 'Baymax · Care Intelligence' : 'System';
    const content = h('div', 'msg-content', text);
    msg.append(h('div', 'msg-author', who), content);
    el.chat.append(msg);
    scrollChat();
    return { msg, content };
  }

  // Never stack identical system messages back-to-back
  const addSystem = (text) => {
    const last = el.chat.lastElementChild;
    if (last && last.classList.contains('system-msg')) {
      const c = last.querySelector('.msg-content');
      if (c && c.textContent === text) { scrollChat(); return null; }
    }
    return addMessage('system-msg', text);
  };

  function sectionLabel(s) {
    return SECTION_LABELS[s] || s;
  }

  function addRobotMessage(text, sources) {
    const m = addMessage('robot', text);
    const uniq = [...new Set((sources || []).map((s) => s.section))];
    if (uniq.length) {
      const foot = h('div', 'msg-sources');
      uniq.forEach((s) => foot.append(h('span', 'chip chip-info', sectionLabel(s))));
      m.msg.append(foot);
      scrollChat();
    }
  }

  // ── Clinical Sources (Audit Trail) ─────────────────────────────────────────
  function renderSources(sources) {
    el.sources.replaceChildren();
    if (!sources || !sources.length) {
      el.sources.append(h('p', 'muted', 'Ask a question or describe symptoms to view cited health records.'));
      return;
    }
    const seen = new Set();
    let idx = 1;
    sources.forEach((s) => {
      const key = (s.text || '').replace(/\s+/g, ' ').trim().toLowerCase();
      if (seen.has(key)) return;
      seen.add(key);
      const item = h('div', 'source-item');
      const top = h('div', 'source-top');
      top.append(
        h('span', 'source-index', '[' + idx++ + ']'),
        h('span', 'chip chip-source', sectionLabel(s.section))
      );
      item.append(top, h('p', 'source-text', s.text));
      el.sources.append(item);
    });
  }

  // ── Executive Patient Summary Card ─────────────────────────────────────────
  function block(title, node) {
    const b = h('div', 'pt-block');
    b.append(h('h3', '', title), node);
    return b;
  }

  function renderOverview(p) {
    const frag = document.createDocumentFragment();

    // 1. Patient Monogram + Header Info
    const initials = (p.name || 'PT').split(' ').map((n) => n[0]).slice(0, 2).join('').toUpperCase();
    const headRow = h('div', 'pt-profile-header');
    const avatar = h('div', 'pt-avatar', initials);
    const info = h('div', 'pt-avatar-info');
    info.append(
      h('div', 'pt-name', p.name || p.id),
      h('div', 'pt-physician', p.primary_doctor ? 'Primary: ' + p.primary_doctor : 'Physician unassigned')
    );
    headRow.append(avatar, info);
    frag.append(headRow);

    // 2. Clinical Vitals & Demographics 3-Card Grid
    const vitalsGrid = h('div', 'pt-vitals-grid');
    const ageTile = h('div', 'vitals-tile');
    ageTile.append(h('span', 'tile-label', 'Age'), h('span', 'tile-val', p.age ? p.age + ' yrs' : 'N/A'));
    const sexTile = h('div', 'vitals-tile');
    sexTile.append(h('span', 'tile-label', 'Sex'), h('span', 'tile-val', p.sex || 'N/A'));
    const bloodTile = h('div', 'vitals-tile blood-tile');
    bloodTile.append(h('span', 'tile-label', 'Blood Type'), h('span', 'tile-val blood-val', p.blood_group || 'N/A'));
    vitalsGrid.append(ageTile, sexTile, bloodTile);
    frag.append(vitalsGrid);

    // 3. Allergies & Contraindications
    const allergyList = (p.allergies || []).map((a) => {
      const tag = h('span', 'chip chip-warn');
      tag.append(h('strong', '', a.substance || a));
      if (a.severity) tag.append(h('span', 'chip-sub', ' (' + a.severity + ')'));
      return tag;
    });
    const allergyWrap = h('div', 'chips');
    if (!allergyList.length) allergyWrap.append(h('span', 'muted', 'No known drug allergies'));
    else allergyList.forEach((t) => allergyWrap.append(t));
    frag.append(block('Allergies & Contraindications', allergyWrap));

    // 4. Diagnosed Conditions
    const condWrap = h('div', 'chips');
    if (!(p.conditions || []).length) condWrap.append(h('span', 'muted', 'No chronic conditions listed'));
    else (p.conditions || []).forEach((c) => condWrap.append(h('span', 'chip chip-neutral', c)));
    frag.append(block('Diagnosed Conditions', condWrap));

    // 5. Active Prescription Regimen
    const medList = h('div', 'pt-med-list');
    (p.medications || []).forEach((m) => {
      const item = h('div', 'pt-med-item');
      const medTop = h('div', 'med-top');
      medTop.append(h('span', 'med-name', m.name));
      if (m.dose) medTop.append(h('span', 'med-dose', m.dose));
      item.append(medTop);
      if (m.frequency) item.append(h('div', 'med-freq', m.frequency));
      medList.append(item);
    });
    if (!medList.children.length) medList.append(h('div', 'muted', 'No active medications'));
    frag.append(block('Active Prescriptions', medList));

    // 6. Next Clinical Appointment
    const appt = p.next_appointment;
    if (appt) {
      const apptCard = h('div', 'pt-appt-card');
      const apptDate = h('div', 'appt-badge', appt.date || 'Scheduled');
      const apptDetails = h('div', 'appt-details');
      apptDetails.append(
        h('div', 'appt-doctor', appt.doctor || 'Clinic Visit'),
        h('div', 'appt-meta', [appt.time, appt.reason, appt.location].filter(Boolean).join(' · '))
      );
      apptCard.append(apptDate, apptDetails);
      frag.append(block('Upcoming Appointment', apptCard));
    }

    // 7. Emergency Contact
    if (p.emergency_contact && p.emergency_contact.name) {
      const ec = p.emergency_contact;
      const ecRow = h('div', 'pt-ec-row');
      ecRow.append(
        h('span', 'ec-label', 'Emergency Contact: '),
        h('span', 'ec-val', ec.name + ' (' + (ec.relation || 'Contact') + ') ' + (ec.phone || ''))
      );
      frag.append(ecRow);
    }

    el.patient.replaceChildren(frag);
  }

  async function loadOverview(id) {
    try {
      const r = await fetch('/api/patients/' + encodeURIComponent(id));
      if (!r.ok) throw new Error('HTTP ' + r.status);
      renderOverview(await r.json());
    } catch (e) {
      el.patient.replaceChildren(h('p', 'muted', 'Could not load patient details.'));
    }
  }

  // ── WebSocket ──────────────────────────────────────────────────────────────
  let pendingAudioMeta = null;
  let pendingUser = null;
  let retryTimer = null;
  let pingTimer = null;
  let lastPongTime = Date.now();
  let reconnectCount = 0;

  function send(obj) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(obj));
      return true;
    }
    return false;
  }

  function scheduleReconnect(delayMs) {
    clearTimeout(retryTimer);
    reconnectCount++;
    const ms = delayMs !== undefined ? delayMs : Math.min(2500, 600 + reconnectCount * 400);
    retryTimer = setTimeout(connect, ms);
  }

  function connect() {
    clearTimeout(retryTimer);
    clearInterval(pingTimer);
    if (state.ws) {
      try {
        state.ws.onopen = state.ws.onclose = state.ws.onerror = state.ws.onmessage = null;
        state.ws.close();
      } catch (e) {}
      state.ws = null;
    }

    setConnection('connecting', 'Connecting…');
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = proto + '://' + location.host + '/ws/web';

    let ws;
    try {
      ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
    } catch (err) {
      console.warn('[WS] Create error:', err);
      scheduleReconnect();
      return;
    }
    state.ws = ws;

    ws.onopen = () => {
      state.connected = true;
      reconnectCount = 0;
      lastPongTime = Date.now();
      setConnection('online', 'Connected');
      clearInterval(pingTimer);
      pingTimer = setInterval(() => {
        if (!state.connected || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
        send({ type: 'ping' });
        // Heartbeat timeout watchdog: if no pong or traffic in 16s, reconnect cleanly
        if (Date.now() - lastPongTime > 16000) {
          console.warn('[WS] Heartbeat timeout — reconnecting...');
          try { ws.close(); } catch (_) {}
          scheduleReconnect(400);
        }
      }, 8000);
    };

    ws.onclose = () => {
      state.connected = false;
      clearInterval(pingTimer);
      if (state.recording) stopRecording(false, true);
      stopPlayback();
      setConvMode(false, true);
      setState('idle');
      setConnection('offline', 'Reconnecting…');
      scheduleReconnect();
    };

    ws.onerror = () => { /* onclose handles retry */ };

    ws.onmessage = (ev) => {
      lastPongTime = Date.now();
      if (typeof ev.data === 'string') {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (e) { return; }
        if (msg.type === 'pong') {
          lastPongTime = Date.now();
          return;
        }
        handleServer(msg);
      } else if (pendingAudioMeta) {
        pendingAudioMeta = null;
        queueAudio(ev.data);
      }
    };
  }

  // Active Mobile Lifecycle Reconnection Handlers
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      if (!state.connected || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
        console.log('[WS] Screen unlocked / tab visible — reconnecting immediately');
        connect();
      } else {
        send({ type: 'ping' });
      }
    }
  });

  window.addEventListener('online', () => {
    console.log('[WS] Network back online — reconnecting');
    connect();
  });

  if (el.wsPill) {
    el.wsPill.style.cursor = 'pointer';
    el.wsPill.title = 'Tap to reconnect';
    el.wsPill.addEventListener('click', () => {
      if (!state.connected || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
        connect();
      }
    });
  }

  function handleServer(msg) {
    switch (msg.type) {
      case 'ready': {
        state.patientId = msg.patient_id;
        el.select.replaceChildren();
        (msg.patients || []).forEach((p) => {
          const o = h('option', '', p.name);
          o.value = p.id;
          el.select.append(o);
        });
        if (state.patientId) el.select.value = state.patientId;
        if (msg.stt && msg.tts) {
          el.engineInfo.textContent =
            'Speech: ' + msg.stt.backend + ' (' + msg.stt.model + '), Voice: ' + msg.tts.voice;
        }
        if (!(msg.patients || []).length) {
          addSystem('No patient records found. Run: python pdf_ingest.py <file.pdf>');
        } else if (state.patientId) {
          loadOverview(state.patientId);
        }
        if (!el.chat.querySelector('.dialogue-msg.robot')) {
          addRobotMessage('Hi, I am Baymax, your personal healthcare assistant. How can I help you today?', []);
        }
        break;
      }
      case 'patient':
        state.patientId = msg.patient_id;
        loadOverview(msg.patient_id);
        addSystem('Switched active record.');
        renderSources([]);
        break;
      case 'status':
        if (!state.recording) setState(msg.state);
        break;
      case 'transcript':
        state.silent = 0;
        if (pendingUser) {
          pendingUser.msg.classList.remove('pending');
          pendingUser.content.textContent = msg.text;
          pendingUser = null;
        } else {
          addMessage('user', msg.text);
        }
        break;
      case 'no_speech':
        if (pendingUser) { pendingUser.msg.remove(); pendingUser = null; }
        addSystem(NO_SPEECH_MSG);
        setState('idle');
        noteSilence();
        break;
      case 'reply':
        addRobotMessage(msg.text, msg.sources);
        if (!msg.greeting) renderSources(msg.sources);
        break;
      case 'audio':
        pendingAudioMeta = msg;
        serverDone = false;
        break;
      case 'done': {
        serverDone = true;
        const gen = playGen;
        playChain.then(() => { if (gen === playGen) maybeFinishSpeaking(); });
        break;
      }
      case 'error':
        addSystem(msg.message);
        if (state.ui !== 'speaking') setState('idle');
        break;
      default:
        break;
    }
  }

  // ── Audio playback (gapless queue of MP3 sentences) ────────────────────────
  let audioCtx = null;
  let playGen = 0;
  let nextStart = 0;
  let playChain = Promise.resolve();
  let serverDone = true;
  const activeSources = new Set();

  async function ensureAudio() {
    if (!audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AC();
    }
    if (audioCtx.state === 'suspended') await audioCtx.resume();
  }

  function queueAudio(arrayBuffer) {
    const gen = playGen;
    playChain = playChain.then(async () => {
      if (gen !== playGen || !audioCtx) return;
      try {
        const buf = await audioCtx.decodeAudioData(arrayBuffer);
        if (gen !== playGen) return;
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(audioCtx.destination);
        const startAt = Math.max(audioCtx.currentTime + 0.03, nextStart);
        src.start(startAt);
        nextStart = startAt + buf.duration;
        activeSources.add(src);
        src.onended = () => {
          activeSources.delete(src);
          if (gen === playGen) maybeFinishSpeaking();
        };
      } catch (e) {
        console.warn('Could not play audio', e);
      }
    });
  }

  function stopPlayback() {
    playGen++;
    nextStart = 0;
    playChain = Promise.resolve();
    pendingAudioMeta = null;
    activeSources.forEach((s) => { try { s.onended = null; s.stop(); } catch (e) { /* already stopped */ } });
    activeSources.clear();
  }

  function maybeFinishSpeaking() {
    if (serverDone && activeSources.size === 0 && state.ui === 'speaking') {
      setState('idle');
      if (state.convMode) setTimeout(autoListen, 450);
    }
  }

  // ── Microphone capture ─────────────────────────────────────────────────────
  const WORKLET_CODE = `
    class PCMCapture extends AudioWorkletProcessor {
      constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch) {
          for (let i = 0; i < ch.length; i++) {
            this.buf[this.n++] = ch[i];
            if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice(0)); this.n = 0; }
          }
        }
        return true;
      }
    }
    registerProcessor('pcm-capture', PCMCapture);
  `;

  let micStream = null, micSource = null, workletNode = null, muteGain = null, workletReady = false;
  let rec = null;

  function makeDownsampler(inRate, outRate) {
    const ratio = inRate / outRate;
    let acc = 0, cnt = 0, seen = 0, boundary = ratio;
    return function (input) {
      const out = [];
      for (let i = 0; i < input.length; i++) {
        acc += input[i];
        cnt++;
        seen++;
        if (seen >= boundary) {
          out.push(acc / cnt);
          acc = 0;
          cnt = 0;
          boundary += ratio;
        }
      }
      const pcm = new Int16Array(out.length);
      for (let i = 0; i < out.length; i++) {
        const v = Math.max(-1, Math.min(1, out[i]));
        pcm[i] = v < 0 ? v * 32768 : v * 32767;
      }
      return pcm;
    };
  }

  function micErrorMessage(e) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return 'Microphone requires HTTPS or localhost. Open this page securely.';
    }
    if (e && (e.name === 'NotAllowedError' || e.name === 'SecurityError')) {
      return 'Microphone permission blocked. Please allow microphone access in your browser settings.';
    }
    if (e && e.name === 'NotFoundError') return 'No microphone hardware found on this device.';
    return 'Microphone error: ' + ((e && e.message) || e);
  }

  async function startRecording() {
    if (state.recording || state.starting) return;
    if (!state.connected) { addSystem('Not connected to the server yet — please wait a moment.'); return; }
    state.starting = true;
    try {
      await ensureAudio();
      stopPlayback();
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      micSource = audioCtx.createMediaStreamSource(micStream);
      muteGain = audioCtx.createGain();
      muteGain.gain.value = 0;

      // Direct, universal ScriptProcessorNode (zero blob/CSP issues, synchronous chunks)
      const sp = audioCtx.createScriptProcessor(2048, 1, 1);
      sp.onaudioprocess = (e) => {
        if (state.recording) {
          onMicChunk(e.inputBuffer.getChannelData(0));
        }
      };

      workletNode = sp;
      micSource.connect(workletNode);
      workletNode.connect(muteGain);
      muteGain.connect(audioCtx.destination);
    } catch (e) {
      state.starting = false;
      addSystem(micErrorMessage(e));
      if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
      return;
    }

    send({ type: 'start' });
    rec = {
      down: makeDownsampler(audioCtx.sampleRate, TARGET_RATE),
      startedAt: performance.now(),
      lastSpeech: 0,
      speech: false,
      noise: [],
      threshold: 0.003,
      totalBytesSent: 0,
    };

    state.recording = true;
    state.starting = false;
    setState('listening');
    el.micHint.textContent = 'Listening… speak now. Tap microphone again to finish.';
  }

  function onMicChunk(f32) {
    if (!state.recording || !rec) return;
    let sum = 0;
    for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
    const rms = Math.sqrt(sum / f32.length);
    const lvl = Math.min(1, rms * 15).toFixed(2);
    el.mic.style.setProperty('--level', lvl);
    if (el.soundbars) {
      el.soundbars.classList.add('active');
      el.soundbars.style.setProperty('--level', lvl);
    }

    const pcm = rec.down(f32);
    if (pcm.length && state.ws && state.ws.readyState === WebSocket.OPEN) {
      rec.totalBytesSent += pcm.byteLength;
      state.ws.send(pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength));
    }

    const now = performance.now();
    if (rec.noise.length < 8) {
      rec.noise.push(rms);
      if (rec.noise.length === 8) {
        rec.threshold = Math.min(0.04, Math.max(0.003, Math.min(...rec.noise) * 2));
      }
    }
    if (rms > rec.threshold) {
      rec.speech = true;
      rec.lastSpeech = now;
    }

    // Snappy conversational pause detection (850ms) or manual finish
    if (rec.speech && now - rec.lastSpeech > 850) {
      stopRecording(true);
    } else if (!rec.speech && now - rec.startedAt > 8000) {
      stopRecording(false);
    } else if (now - rec.startedAt > 30000) {
      stopRecording(true);
    }
  }

  function stopRecording(sendEnd, silent) {
    if (!state.recording) return;
    state.recording = false;
    const sentBytes = rec ? rec.totalBytesSent : 0;
    try {
      if (workletNode) {
        if (workletNode.onaudioprocess) workletNode.onaudioprocess = null;
        workletNode.disconnect();
      }
      if (micSource) micSource.disconnect();
      if (muteGain) muteGain.disconnect();
    } catch (e) { /* already disconnected */ }
    if (micStream) micStream.getTracks().forEach((t) => t.stop());
    micStream = micSource = workletNode = muteGain = null;
    rec = null;
    el.mic.style.removeProperty('--level');
    if (el.soundbars) {
      el.soundbars.classList.remove('active');
      el.soundbars.style.removeProperty('--level');
    }
    el.micHint.textContent = 'Tap microphone to speak, or type below. Auto-detects speech.';
    if (silent) { setState('idle'); return; }

    if (sendEnd || sentBytes >= 3200) {
      send({ type: 'end' });
      pendingUser = addMessage('user', 'Listening…', true);
      setState('transcribing');
    } else {
      send({ type: 'cancel' });
      setState('idle');
      addSystem(NO_SPEECH_MSG);
      noteSilence();
    }
  }

  // ── Text input & controls ──────────────────────────────────────────────────
  async function sendText(text) {
    text = (text === undefined ? el.input.value : text).trim();
    if (!text) return;
    if (!state.connected) { addSystem('Not connected to the server yet — please wait a moment.'); return; }
    if (state.recording) stopRecording(false, true);
    try { await ensureAudio(); } catch (e) { /* audio playback may be blocked */ }
    stopPlayback();
    addMessage('user', text);
    el.input.value = '';
    send({ type: 'text', text });
    setState('thinking');
  }

  el.mic.addEventListener('click', () => {
    if (state.recording) stopRecording(true);
    else startRecording();
  });
  el.send.addEventListener('click', () => sendText());
  el.input.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendText(); });
  el.suggestions.addEventListener('click', (e) => {
    const b = e.target.closest('.chip-btn');
    if (b) {
      const q = b.getAttribute('data-query') || b.querySelector('.chip-label')?.textContent || b.textContent;
      sendText(q);
    }
  });
  el.select.addEventListener('change', () => {
    if (state.recording) stopRecording(false, true);
    stopPlayback();
    send({ type: 'patient', id: el.select.value });
  });

  // Mobile View Switcher (Clinical Dialogue vs. Patient Record)
  const tabDialogue = document.getElementById('tabDialogueBtn');
  const tabPatient = document.getElementById('tabPatientBtn');
  const chatCardEl = document.querySelector('.chat-card');
  const sideColEl = document.querySelector('.side-col');

  if (tabDialogue && tabPatient) {
    tabDialogue.addEventListener('click', () => {
      tabDialogue.classList.add('active');
      tabDialogue.setAttribute('aria-selected', 'true');
      tabPatient.classList.remove('active');
      tabPatient.setAttribute('aria-selected', 'false');
      chatCardEl?.classList.remove('mobile-view-hidden');
      sideColEl?.classList.add('mobile-view-hidden');
    });

    tabPatient.addEventListener('click', () => {
      tabPatient.classList.add('active');
      tabPatient.setAttribute('aria-selected', 'true');
      tabDialogue.classList.remove('active');
      tabDialogue.setAttribute('aria-selected', 'false');
      sideColEl?.classList.remove('mobile-view-hidden');
      chatCardEl?.classList.add('mobile-view-hidden');
    });
  }

  // ── Conversation mode: hands-free back-and-forth ───────────────────────────
  function noteSilence() {
    if (!state.convMode) return;
    state.silent += 1;
    if (state.silent >= 2) {
      setConvMode(false, true);
      addSystem("Standing by. Turn on continuous listening or tap the microphone when ready.");
    } else {
      setTimeout(autoListen, 300);
    }
  }

  function autoListen() {
    if (!state.convMode || state.recording || state.starting || !state.connected) return;
    if (state.ui !== 'idle') return;
    startRecording();
  }

  async function setConvMode(on, silently) {
    state.convMode = on;
    state.silent = 0;
    el.convBtn.classList.toggle('on', on);
    el.convBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    el.convBtn.textContent = on ? 'Continuous listening: On' : 'Continuous listening: Off';
    if (on) {
      if (!state.connected) {
        setConvMode(false, true);
        addSystem('Not connected to the server yet — please wait a moment.');
        return;
      }
      try { await ensureAudio(); } catch (e) { /* playback may be blocked */ }
      if (el.chat.querySelector('.dialogue-msg.user')) autoListen();
      else send({ type: 'greet' });
    } else if (!silently) {
      if (state.recording) stopRecording(false, true);
      stopPlayback();
      send({ type: 'cancel' });
      setState('idle');
    }
  }

  function newConversation() {
    if (state.recording) stopRecording(false, true);
    stopPlayback();
    send({ type: 'reset' });
    pendingUser = null;
    el.chat.replaceChildren();
    renderSources([]);
    setState('idle');
    addSystem('Started a new conversation.');
    if (state.convMode) send({ type: 'greet' });
  }

  el.convBtn.addEventListener('click', () => setConvMode(!state.convMode));
  el.newChat.addEventListener('click', newConversation);

  addSystem('Session ready. Tap the microphone or type below. Turn on continuous listening for hands-free triage.');
  setState('idle');
  connect();
})();

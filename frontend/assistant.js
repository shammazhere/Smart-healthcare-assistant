/**
 * HealthBot Assistant — voice + chat page
 * Talks to the AI gateway (server.py) over WebSocket /ws/web:
 *   mic audio (16 kHz PCM) or typed text  ->  records search + Groq  ->  text reply + spoken MP3
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const el = {
    pill: $('wsPill'), pillText: $('wsStatusText'), select: $('patientSelect'),
    engineInfo: $('engineInfo'), statePill: $('audioStatePill'), stateText: $('audioStateText'),
    chat: $('chatScroll'), suggestions: $('suggestions'), mic: $('micBtn'), micHint: $('micHint'),
    input: $('textInput'), send: $('sendBtn'), patient: $('patientCard'), sources: $('sourcesList'),
    convBtn: $('convBtn'), newChat: $('newChatBtn'),
  };

  const TARGET_RATE = 16000;
  const NO_SPEECH_MSG = "I didn't hear anything. Tap the microphone and try again.";
  const STATE_LABELS = {
    idle: 'Idle', listening: 'Listening…', transcribing: 'Understanding…',
    thinking: 'Checking records…', speaking: 'Speaking…',
  };
  const SECTION_LABELS = {
    profile: 'Profile', allergies: 'Allergy', conditions: 'Condition', surgeries: 'Surgery',
    medications: 'Medicine', lab_results: 'Lab result', vitals_baseline: 'Usual vitals',
    visits: 'Doctor visit', appointments: 'Appointment', care_instructions: 'Care instruction',
    doctor_notes: "Doctor's note",
  };

  const state = { ws: null, connected: false, patientId: null, ui: 'idle', recording: false, starting: false, convMode: false, silent: 0 };

  // ── tiny DOM helper ────────────────────────────────────────────────────────
  function h(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  // ── UI state ───────────────────────────────────────────────────────────────
  function setState(s) {
    state.ui = s;
    el.stateText.textContent = STATE_LABELS[s] || s;
    const cls = s === 'transcribing' ? 'thinking' : s;
    el.statePill.className = 'audio-state-pill' + (cls !== 'idle' ? ' ' + cls : '');
    el.mic.classList.toggle('active', state.recording);
  }

  function setConnection(kind, label) {
    el.pill.className = 'status-pill ' + kind;
    el.pillText.textContent = label;
  }

  // Scrolls only the chat box (never the whole page)
  function scrollChat() {
    el.chat.scrollTop = el.chat.scrollHeight;
  }

  function addMessage(kind, text, pending) {
    const msg = h('div', 'dialogue-msg ' + kind + (pending ? ' pending' : ''));
    const who = kind === 'user' ? 'You' : kind === 'robot' ? 'HealthBot' : 'System';
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

  // Replaces the list every time and skips repeated records
  function renderSources(sources) {
    el.sources.replaceChildren();
    if (!sources || !sources.length) {
      el.sources.append(h('p', 'muted', 'No specific records were needed for that answer.'));
      return;
    }
    const seen = new Set();
    sources.forEach((s) => {
      const key = (s.text || '').replace(/\s+/g, ' ').trim().toLowerCase();
      if (seen.has(key)) return;
      seen.add(key);
      const item = h('div', 'source-item');
      item.append(h('span', 'chip chip-info', sectionLabel(s.section)), h('p', '', s.text));
      el.sources.append(item);
    });
  }

  // ── Patient card ───────────────────────────────────────────────────────────
  function block(title, node) {
    const b = h('div', 'pt-block');
    b.append(h('h3', '', title), node);
    return b;
  }

  function chips(items, cls) {
    const wrap = h('div', 'chips');
    if (!items.length) wrap.append(h('span', 'muted', 'None recorded'));
    items.forEach((t) => wrap.append(h('span', 'chip ' + (cls || ''), t)));
    return wrap;
  }

  function renderOverview(p) {
    const frag = document.createDocumentFragment();
    const meta = [p.age ? p.age + ' years' : '', p.sex, p.blood_group ? 'Blood group ' + p.blood_group : '']
      .filter(Boolean).join(' • ');
    frag.append(h('div', 'pt-name', p.name || p.id), h('div', 'pt-meta', meta));
    if (p.primary_doctor) frag.append(h('div', 'pt-meta', 'Doctor: ' + p.primary_doctor));

    frag.append(block('Allergies', chips(
      (p.allergies || []).map((a) => (a.severity ? a.substance + ' (' + a.severity + ')' : a.substance)), 'chip-warn')));
    frag.append(block('Conditions', chips(p.conditions || [], '')));

    const ul = h('ul');
    (p.medications || []).forEach((m) => {
      const li = h('li', '', m.name + (m.dose ? ' ' + m.dose : ''));
      li.append(h('span', '', m.frequency || ''));
      ul.append(li);
    });
    if (!ul.children.length) ul.append(h('li', 'muted', 'None recorded'));
    frag.append(block('Medicines', ul));

    const appt = p.next_appointment;
    const apptNode = h('div');
    if (appt) {
      const li = h('li', '', (appt.date || '') + (appt.time ? ' at ' + appt.time : '') + (appt.doctor ? ' — ' + appt.doctor : ''));
      li.append(h('span', '', appt.reason || ''));
      const list = h('ul');
      list.append(li);
      apptNode.append(list);
    } else {
      apptNode.append(h('span', 'muted', 'No upcoming appointment'));
    }
    frag.append(block('Next appointment', apptNode));
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

  function send(obj) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(obj));
      return true;
    }
    return false;
  }

  function connect() {
    clearTimeout(retryTimer);
    setConnection('connecting', 'Connecting…');
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(proto + '://' + location.host + '/ws/web');
    ws.binaryType = 'arraybuffer';
    state.ws = ws;

    ws.onopen = () => {
      state.connected = true;
      setConnection('online', 'Connected');
      clearInterval(pingTimer);
      pingTimer = setInterval(() => send({ type: 'ping' }), 20000);
    };
    ws.onclose = () => {
      state.connected = false;
      clearInterval(pingTimer);
      if (state.recording) stopRecording(false, true);
      stopPlayback();
      setConvMode(false, true);
      setState('idle');
      setConnection('offline', 'Disconnected — retrying…');
      retryTimer = setTimeout(connect, 2000);
    };
    ws.onerror = () => { /* onclose handles the retry */ };
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (e) { return; }
        handleServer(msg);
      } else if (pendingAudioMeta) {
        pendingAudioMeta = null;
        queueAudio(ev.data);
      }
    };
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
            'Speech: ' + msg.stt.backend + ' ' + msg.stt.model + ' • Voice: ' + msg.tts.voice;
        }
        if (!(msg.patients || []).length) {
          addSystem('No patient records found. Run: python pdf_ingest.py <file.pdf>');
        } else if (state.patientId) {
          loadOverview(state.patientId);
        }
        break;
      }
      case 'patient':
        state.patientId = msg.patient_id;
        loadOverview(msg.patient_id);
        addSystem('Switched patient.');
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
        addSystem('⚠️ ' + msg.message);
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
      if (state.convMode) setTimeout(autoListen, 450);   // conversation mode: listen again
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

  // <downsampler>
  // Box-filter downsampler: Float32 at any rate -> Int16 at TARGET_RATE (keeps state between chunks).
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
  // </downsampler>

  function micErrorMessage(e) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return 'The microphone needs a secure page. Open this page as http://localhost:8765 on the same computer (or use HTTPS).';
    }
    if (e && (e.name === 'NotAllowedError' || e.name === 'SecurityError')) {
      return 'Microphone access was blocked. Click the lock icon in the address bar, allow the microphone, and try again.';
    }
    if (e && e.name === 'NotFoundError') return 'No microphone was found on this device.';
    return 'Could not start the microphone: ' + ((e && e.message) || e);
  }

  async function startRecording() {
    if (state.recording || state.starting) return;      // ignore double taps while starting
    if (!state.connected) { addSystem('Not connected to the server yet — please wait a moment.'); return; }
    state.starting = true;
    try {
      await ensureAudio();
      stopPlayback();
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      if (!workletReady) {
        const url = URL.createObjectURL(new Blob([WORKLET_CODE], { type: 'application/javascript' }));
        await audioCtx.audioWorklet.addModule(url);
        workletReady = true;
      }
    } catch (e) {
      state.starting = false;
      addSystem(micErrorMessage(e));
      if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
      return;
    }

    send({ type: 'start' });
    rec = {
      down: makeDownsampler(audioCtx.sampleRate, TARGET_RATE),
      startedAt: performance.now(), lastSpeech: 0, speech: false, noise: [], threshold: 0.02,
    };
    micSource = audioCtx.createMediaStreamSource(micStream);
    workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture');
    muteGain = audioCtx.createGain();
    muteGain.gain.value = 0;                       // keeps the graph running without echoing the mic
    micSource.connect(workletNode);
    workletNode.connect(muteGain);
    muteGain.connect(audioCtx.destination);
    workletNode.port.onmessage = (e) => onMicChunk(e.data);

    state.recording = true;
    state.starting = false;
    setState('listening');
    el.micHint.textContent = 'Listening… speak now. Tap the microphone again to stop.';
  }

  function onMicChunk(f32) {
    if (!state.recording || !rec) return;
    let sum = 0;
    for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
    const rms = Math.sqrt(sum / f32.length);
    el.mic.style.setProperty('--level', Math.min(1, rms * 8).toFixed(2));

    const pcm = rec.down(f32);
    if (pcm.length && state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(pcm.buffer);

    const now = performance.now();
    if (rec.noise.length < 6) {                     // learn the room's background level
      rec.noise.push(rms);
      if (rec.noise.length === 6) rec.threshold = Math.min(0.05, Math.max(0.012, Math.min(...rec.noise) * 3));
    }
    if (rms > rec.threshold) { rec.speech = true; rec.lastSpeech = now; }

    if (rec.speech && now - rec.lastSpeech > 1400) stopRecording(true);        // finished talking
    else if (!rec.speech && now - rec.startedAt > 8000) stopRecording(false);   // nobody spoke
    else if (now - rec.startedAt > 30000) stopRecording(true);                  // safety limit
  }

  function stopRecording(sendEnd, silent) {
    if (!state.recording) return;
    state.recording = false;
    try {
      workletNode.port.onmessage = null;
      micSource.disconnect();
      workletNode.disconnect();
      muteGain.disconnect();
    } catch (e) { /* already disconnected */ }
    if (micStream) micStream.getTracks().forEach((t) => t.stop());
    micStream = micSource = workletNode = muteGain = null;
    rec = null;
    el.mic.style.removeProperty('--level');
    el.micHint.textContent = 'Tap the microphone and speak. It stops by itself when you finish talking.';
    if (silent) { setState('idle'); return; }

    if (sendEnd) {
      send({ type: 'end' });
      pendingUser = addMessage('user', '🎙️ …', true);
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
    try { await ensureAudio(); } catch (e) { /* playback may be blocked; text still works */ }
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
    if (b) sendText(b.textContent);
  });
  el.select.addEventListener('change', () => {
    if (state.recording) stopRecording(false, true);
    stopPlayback();
    send({ type: 'patient', id: el.select.value });
  });

  // ── Start ──────────────────────────────────────────────────────────────────
  // ── Conversation mode: hands-free back-and-forth ───────────────────────────
  function noteSilence() {
    if (!state.convMode) return;
    state.silent += 1;
    if (state.silent >= 2) {
      setConvMode(false, true);
      addSystem("I'll wait here. Turn on conversation mode or tap the microphone when you want to talk.");
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
    el.convBtn.textContent = on ? '🗣️ Conversation mode: On' : '🗣️ Conversation mode: Off';
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

  addSystem('Hello! Tap the microphone or type to talk. Turn on conversation mode for a hands-free chat.');
  setState('idle');
  connect();
})();
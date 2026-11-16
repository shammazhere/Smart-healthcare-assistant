/**
 * HealthBot OS — Web Dashboard Controller
 * Connects to ESP32-S3 via WebSocket for Telemetry, Vitals & Motor Control.
 */

// ── Application State ────────────────────────────────────────────────────────
const state = {
  ws: null,
  robotIp: localStorage.getItem('healthbot_ip') || (window.location.host ? window.location.host : 'localhost:8765'),
  connected: false,
  voiceActive: false,
  activeDirection: null,
  driveInterval: null,
  speed: 170,
  pingStartTime: 0,
  latency: null,
  startTime: Date.now(),
  ecgPhase: 0,
  lastBeatTime: 0,
  currentHR: 0,
  currentSpO2: 0,
  fingerOk: false,
};

// ── DOM Elements ─────────────────────────────────────────────────────────────
const elements = {
  robotIpInput: document.getElementById('robotIpInput'),
  connectBtn: document.getElementById('connectBtn'),
  wsPill: document.getElementById('wsPill'),
  wsStatusText: document.getElementById('wsStatusText'),
  latencyText: document.getElementById('latencyText'),
  fingerBadge: document.getElementById('fingerBadge'),
  fingerText: document.getElementById('fingerText'),
  hrValue: document.getElementById('hrValue'),
  pulseIndicator: document.getElementById('pulseIndicator'),
  hrCategory: document.getElementById('hrCategory'),
  spo2Value: document.getElementById('spo2Value'),
  spo2Bar: document.getElementById('spo2Bar'),
  spo2Category: document.getElementById('spo2Category'),
  ecgCanvas: document.getElementById('ecgCanvas'),
  audioStatePill: document.getElementById('audioStatePill'),
  audioStateText: document.getElementById('audioStateText'),
  voiceMicBtn: document.getElementById('voiceMicBtn'),
  voiceHint: document.getElementById('voiceHint'),
  dialogueScroll: document.getElementById('dialogueScroll'),
  textPromptInput: document.getElementById('textPromptInput'),
  speedSlider: document.getElementById('speedSlider'),
  speedDisplay: document.getElementById('speedDisplay'),
  uptimeBadge: document.getElementById('uptimeBadge'),
  dpadButtons: document.querySelectorAll('.dpad-btn'),
};

// ── Initialize App ───────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  elements.robotIpInput.value = state.robotIp;
  initCanvas();
  bindDriveControls();
  bindKeyboardControls();
  startUptimeTimer();

  // If already loaded from ESP32 or configured IP, attempt auto-connect
  if (state.robotIp && state.robotIp !== '') {
    connectWebSocket();
  }
});

// ── WebSocket Connection Manager ─────────────────────────────────────────────
function toggleConnection() {
  if (state.connected) {
    disconnectWebSocket();
  } else {
    state.robotIp = elements.robotIpInput.value.trim();
    localStorage.setItem('healthbot_ip', state.robotIp);
    connectWebSocket();
  }
}

function connectWebSocket() {
  const ip = elements.robotIpInput.value.trim() || window.location.host || 'localhost:8765';
  const url = `ws://${ip}/ws/control`;

  updateConnectionStatus('connecting', 'Connecting...');
  
  try {
    if (state.ws) {
      state.ws.close();
    }

    state.ws = new WebSocket(url);

    state.ws.onopen = () => {
      state.connected = true;
      state.startTime = Date.now();
      updateConnectionStatus('online', 'Connected');
      addTranscriptMessage('system', `Connected to HealthBot at ${ip}`);
      measurePing();
    };

    state.ws.onclose = () => {
      state.connected = false;
      updateConnectionStatus('offline', 'Disconnected');
      resetVitalsDisplay();
    };

    state.ws.onerror = (err) => {
      console.warn('[WS Error]', err);
      state.connected = false;
      updateConnectionStatus('offline', 'Error');
    };

    state.ws.onmessage = (event) => {
      handleIncomingTelemetry(event.data);
    };

  } catch (err) {
    console.error('Failed to create WebSocket:', err);
    updateConnectionStatus('offline', 'Failed');
  }
}

function disconnectWebSocket() {
  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }
  state.connected = false;
  updateConnectionStatus('offline', 'Disconnected');
}

function updateConnectionStatus(type, label) {
  elements.wsStatusText.textContent = label;
  elements.wsPill.className = `status-pill ${type}`;

  if (type === 'online') {
    elements.connectBtn.classList.add('connected');
    elements.connectBtn.querySelector('.btn-label').textContent = 'Disconnect';
  } else {
    elements.connectBtn.classList.remove('connected');
    elements.connectBtn.querySelector('.btn-label').textContent = 'Connect';
    elements.latencyText.textContent = '-- ms';
  }
}

function sendWs(data) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(typeof data === 'string' ? data : JSON.stringify(data));
  }
}

// ── Telemetry & Vitals Handler ───────────────────────────────────────────────
function handleIncomingTelemetry(raw) {
  try {
    const data = JSON.parse(raw);

    // Latency calculation
    if (state.pingStartTime > 0) {
      const now = performance.now();
      state.latency = Math.round(now - state.pingStartTime);
      elements.latencyText.textContent = `${state.latency} ms`;
      state.pingStartTime = 0;
    }

    // Finger Presence
    if (data.finger !== undefined) {
      state.fingerOk = Boolean(data.finger);
      if (state.fingerOk) {
        elements.fingerBadge.classList.add('active');
        elements.fingerText.textContent = 'Finger Detected';
      } else {
        elements.fingerBadge.classList.remove('active');
        elements.fingerText.textContent = 'Place Finger on Sensor';
        resetVitalsDisplay();
      }
    }

    // Heart Rate
    if (data.hr !== undefined && state.fingerOk && data.hr > 0) {
      state.currentHR = Math.round(data.hr);
      elements.hrValue.textContent = state.currentHR;
      elements.hrCategory.textContent = getHeartRateCategory(state.currentHR);

      // Trigger beat pulse
      triggerBeatAnimation(state.currentHR);
    }

    // SpO2
    if (data.spo2 !== undefined && state.fingerOk && data.spo2 > 0) {
      state.currentSpO2 = parseFloat(data.spo2.toFixed(1));
      elements.spo2Value.textContent = state.currentSpO2;
      
      const pct = Math.max(0, Math.min(100, ((state.currentSpO2 - 70) / 30) * 100));
      elements.spo2Bar.style.width = `${pct}%`;
      elements.spo2Category.textContent = getSpO2Category(state.currentSpO2);
    }

    // Audio Pipeline State (0: IDLE, 1: LISTENING, 2: SPEAKING)
    if (data.audio !== undefined) {
      updateAudioState(data.audio);
    }

    // AI Transcript Message
    if (data.transcript) {
      addTranscriptMessage('robot', data.transcript);
      speakReply(data.transcript);
    }

  } catch (err) {
    // Non-JSON message
    console.debug('Raw WS:', raw);
  }
}

function resetVitalsDisplay() {
  elements.hrValue.textContent = '--';
  elements.hrCategory.textContent = 'No Finger Detected';
  elements.spo2Value.textContent = '--';
  elements.spo2Bar.style.width = '0%';
  elements.spo2Category.textContent = 'Awaiting signal';
  elements.pulseIndicator.classList.remove('beating');
}

function getHeartRateCategory(bpm) {
  if (bpm < 60) return 'Bradycardia (Low)';
  if (bpm <= 100) return 'Normal Resting Pulse';
  if (bpm <= 130) return 'Elevated Heart Rate';
  return 'Tachycardia (High)';
}

function getSpO2Category(spo2) {
  if (spo2 >= 95) return 'Optimal Oxygenation (Normal)';
  if (spo2 >= 90) return 'Acceptable Oxygenation';
  return 'Hypoxia Alert (< 90%)';
}

function triggerBeatAnimation(bpm) {
  const p = elements.pulseIndicator;
  p.classList.add('beating');
  const duration = (60 / bpm).toFixed(2);
  p.style.animationDuration = `${duration}s`;
}

// ── Web Speech Recognition & Audio Capture ──────────────────────────────────
let recognition = null;
const speechSynth = window.speechSynthesis || null;

function initSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    console.warn('[Speech] Web Speech API not supported in this browser.');
    return null;
  }
  const sr = new SpeechRec();
  sr.continuous = false;
  sr.interimResults = true;
  sr.lang = 'en-US';

  sr.onstart = () => {
    state.voiceActive = true;
    elements.voiceMicBtn.classList.add('active');
    updateAudioState(1);
    elements.voiceHint.textContent = 'Listening… speak your symptom or question.';
    sendWs({ voice: true });
  };

  sr.onresult = (event) => {
    let interim = '';
    let final = '';
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        final += event.results[i][0].transcript;
      } else {
        interim += event.results[i][0].transcript;
      }
    }

    if (interim) {
      elements.voiceHint.textContent = `Hearing: "${interim}"`;
    }

    if (final && final.trim()) {
      const recognizedText = final.trim();
      elements.voiceHint.textContent = `Captured: "${recognizedText}"`;
      addTranscriptMessage('user', recognizedText);
      sendWs({ query: recognizedText });
      updateAudioState(2);
    }
  };

  sr.onerror = (event) => {
    console.warn('[Speech Recognition Error]', event.error);
    if (event.error === 'not-allowed') {
      addTranscriptMessage('system', 'Microphone permission blocked. Please allow mic access in your browser address bar.');
    } else if (event.error !== 'no-speech') {
      addTranscriptMessage('system', `Voice input error: ${event.error}`);
    }
    stopVoiceRecognition();
  };

  sr.onend = () => {
    stopVoiceRecognition();
  };

  return sr;
}

function stopVoiceRecognition() {
  state.voiceActive = false;
  elements.voiceMicBtn.classList.remove('active');
  updateAudioState(0);
  elements.voiceHint.textContent = 'Tap to speak with HealthBot';
  sendWs({ voice: false });
}

function speakReply(text) {
  if (!speechSynth) return;
  try {
    speechSynth.cancel();
    const cleanText = text.replace(/[*_#`]/g, '').trim();
    const utt = new SpeechSynthesisUtterance(cleanText);
    utt.rate = 1.0;
    utt.pitch = 1.0;
    
    // Pick an English or natural voice if available
    const voices = speechSynth.getVoices();
    const enVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Neural'))) || voices.find(v => v.lang.startsWith('en'));
    if (enVoice) utt.voice = enVoice;

    utt.onstart = () => updateAudioState(2);
    utt.onend = () => updateAudioState(0);
    utt.onerror = () => updateAudioState(0);

    speechSynth.speak(utt);
  } catch (e) {
    console.warn('Speech synthesis error:', e);
  }
}

// ── Voice Assistant Controls ─────────────────────────────────────────────────
function toggleVoice() {
  if (state.voiceActive) {
    if (recognition) {
      try { recognition.stop(); } catch (e) {}
    }
    stopVoiceRecognition();
    return;
  }

  if (!recognition) {
    recognition = initSpeechRecognition();
  }

  if (recognition) {
    try {
      recognition.start();
    } catch (e) {
      console.warn('Recognition start error:', e);
      try { recognition.stop(); } catch (_) {}
      setTimeout(() => { try { recognition.start(); } catch (_) {} }, 200);
    }
  } else {
    addTranscriptMessage('system', 'Speech recognition not available. Please use the Baymax Assistant page or type in the input box.');
  }
}

function updateAudioState(stateNum) {
  const pill = elements.audioStatePill;
  const txt = elements.audioStateText;

  pill.className = 'audio-state-pill';

  if (stateNum === 1) {
    pill.classList.add('listening');
    txt.textContent = 'Listening...';
    elements.voiceMicBtn.classList.add('active');
  } else if (stateNum === 2) {
    pill.classList.add('speaking');
    txt.textContent = 'Speaking (TTS)';
  } else {
    txt.textContent = 'Idle';
    if (!state.voiceActive) {
      elements.voiceMicBtn.classList.remove('active');
    }
  }
}

function sendTextPrompt() {
  const input = elements.textPromptInput;
  const text = input.value.trim();
  if (!text) return;

  addTranscriptMessage('user', text);
  input.value = '';

  // Send query command over WS
  sendWs({ query: text });
}

function addTranscriptMessage(sender, text) {
  const scroll = elements.dialogueScroll;
  const msg = document.createElement('div');
  msg.className = `dialogue-msg ${sender}`;

  const author = document.createElement('div');
  author.className = 'msg-author';
  author.textContent = sender === 'user' ? 'Patient' : sender === 'robot' ? 'HealthBot AI' : 'System';

  const content = document.createElement('div');
  content.className = 'msg-content';
  content.textContent = text;

  msg.appendChild(author);
  msg.appendChild(content);
  scroll.appendChild(msg);

  scroll.scrollTop = scroll.scrollHeight;
}

function clearTranscript() {
  elements.dialogueScroll.innerHTML = '';
  addTranscriptMessage('system', 'Transcript cleared.');
}

// ── Mobility / L298N Motor Driver ─────────────────────────────────────────────
function bindDriveControls() {
  elements.dpadButtons.forEach(btn => {
    const dir = btn.dataset.dir;

    const start = (e) => {
      e.preventDefault();
      startDrive(dir, btn);
    };

    const stop = (e) => {
      e.preventDefault();
      stopDrive(btn);
    };

    btn.addEventListener('mousedown', start);
    btn.addEventListener('mouseup', stop);
    btn.addEventListener('mouseleave', stop);

    btn.addEventListener('touchstart', start, { passive: false });
    btn.addEventListener('touchend', stop, { passive: false });
  });
}

function bindKeyboardControls() {
  const keyMap = {
    'ArrowUp': 'F', 'KeyW': 'F',
    'ArrowDown': 'B', 'KeyS': 'B',
    'ArrowLeft': 'L', 'KeyA': 'L',
    'ArrowRight': 'R', 'KeyD': 'R',
    'Space': 'S'
  };

  window.addEventListener('keydown', (e) => {
    // Avoid interfering with typing in inputs
    if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

    const dir = keyMap[e.code];
    if (dir && state.activeDirection !== dir) {
      e.preventDefault();
      const btn = document.querySelector(`.dpad-btn[data-dir="${dir}"]`);
      startDrive(dir, btn);
    }
  });

  window.addEventListener('keyup', (e) => {
    if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

    const dir = keyMap[e.code];
    if (dir && state.activeDirection === dir) {
      e.preventDefault();
      const btn = document.querySelector(`.dpad-btn[data-dir="${dir}"]`);
      stopDrive(btn);
    }
  });
}

function startDrive(dir, btnEl) {
  if (state.activeDirection === dir) return;
  state.activeDirection = dir;

  if (btnEl) btnEl.classList.add('active');

  const send = () => {
    sendWs({ dir: dir, speed: state.speed });
  };

  send();

  // Continually send command every 150ms to satisfy ESP32 500ms watchdog
  clearInterval(state.driveInterval);
  if (dir !== 'S') {
    state.driveInterval = setInterval(send, 150);
  }
}

function stopDrive(btnEl) {
  clearInterval(state.driveInterval);
  state.driveInterval = null;

  if (state.activeDirection) {
    if (btnEl) btnEl.classList.remove('active');
    state.activeDirection = null;
    sendWs({ dir: 'S', speed: 0 });
  }
}

function onSpeedChange(val) {
  state.speed = parseInt(val, 10);
  const pct = Math.round((state.speed / 255) * 100);
  elements.speedDisplay.textContent = `${pct}% (${state.speed}/255)`;

  // If robot is currently driving, update active speed
  if (state.activeDirection && state.activeDirection !== 'S') {
    sendWs({ dir: state.activeDirection, speed: state.speed });
  }
}

function setSpeed(val) {
  elements.speedSlider.value = val;
  onSpeedChange(val);

  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
}

function sendEmergencyStop() {
  stopDrive();
  sendWs({ dir: 'S', speed: 0 });
  addTranscriptMessage('system', '🚨 Emergency Brake Sent to L298N.');
}

function pingRobot() {
  state.pingStartTime = performance.now();
  sendWs('ping');
}

function measurePing() {
  setInterval(() => {
    if (state.connected && state.pingStartTime === 0) {
      state.pingStartTime = performance.now();
      sendWs('ping');
    }
  }, 4000);
}

// ── ECG / PPG Rhythm Waveform Canvas Animation ───────────────────────────────
function initCanvas() {
  const canvas = elements.ecgCanvas;
  const ctx = canvas.getContext('2d');
  
  // Set real canvas pixel dimensions
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  let step = 0;
  const buffer = new Array(Math.floor(rect.width)).fill(rect.height / 2);

  function drawECG() {
    requestAnimationFrame(drawECG);

    const w = rect.width;
    const h = rect.height;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Synthesize realistic PPG wave based on current HR
    let sample = midY;
    if (state.fingerOk && state.currentHR > 30) {
      const freq = (state.currentHR / 60) * 0.05;
      step += freq;
      const mod = step % (Math.PI * 2);

      // P-Q-R-S-T pulse profile
      if (mod > 0 && mod < 0.4) {
        sample = midY - Math.sin(mod * Math.PI / 0.4) * 8; // P wave
      } else if (mod >= 0.4 && mod < 0.5) {
        sample = midY + 4; // Q dip
      } else if (mod >= 0.5 && mod < 0.7) {
        sample = midY - Math.sin((mod - 0.5) * Math.PI / 0.2) * 32; // R peak!
      } else if (mod >= 0.7 && mod < 0.85) {
        sample = midY + 7; // S dip
      } else if (mod >= 0.85 && mod < 1.4) {
        sample = midY - Math.sin((mod - 0.85) * Math.PI / 0.55) * 12; // T wave
      } else {
        sample = midY + (Math.random() - 0.5) * 1.5; // baseline noise
      }
    } else {
      sample = midY + (Math.random() - 0.5) * 2;
    }

    buffer.shift();
    buffer.push(sample);

    // Render glow path
    ctx.beginPath();
    ctx.strokeStyle = state.fingerOk ? '#34d399' : '#64748b';
    ctx.lineWidth = 2;
    ctx.shadowColor = state.fingerOk ? 'rgba(52, 211, 153, 0.6)' : 'transparent';
    ctx.shadowBlur = 8;

    for (let i = 0; i < buffer.length; i++) {
      if (i === 0) ctx.moveTo(i, buffer[i]);
      else ctx.lineTo(i, buffer[i]);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  drawECG();
}

// ── Uptime Timer ─────────────────────────────────────────────────────────────
function startUptimeTimer() {
  setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - state.startTime) / 1000);
    const mins = String(Math.floor(elapsedSeconds / 60)).padStart(2, '0');
    const secs = String(elapsedSeconds % 60).padStart(2, '0');
    elements.uptimeBadge.textContent = `Up ${mins}:${secs}`;
  }, 1000);
}

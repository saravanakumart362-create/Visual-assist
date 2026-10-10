// The live camera demo: the visitor's camera in place of the simulated room.
// The converted models (demo/models) run here with ONNX Runtime; what they
// find is handed to the project's own assistant (visual_assist/live.py),
// running in the page's Python, which decides what to say.
(() => {
  "use strict";

  const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
  const MODELS = "demo/models/";
  const THUMBNAIL = { width: 32, height: 24 }; // a tiny grey copy, to notice movement
  const NOTE_PICTURES = 3; // a banknote is read from this many pictures, averaged

  const shared = window.visualAssist;
  const $ = (id) => document.getElementById(id);
  const video = $("camera");
  const overlay = $("boxes");
  const toggle = $("camera-toggle");
  const transcript = $("live-transcript");
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const state = {
    stream: null,
    running: 0, // a new number for each start; 0 while the camera is off
    loading: null, // the promise of models and assistant, made once
    detectors: [],
    currency: null,
    iou: 0.7,
    live: null, // the Python assistant
    pending: [], // questions waiting for the next picture
    busyUntil: 0,
    voiceMissing: "",
  };

  const setStatus = (text) => { $("live-status").textContent = text; };

  // ---------------------------------------------------------------- speaking

  function addLine(who, text, language) {
    const item = document.createElement("li");
    item.className = who;
    const name = document.createElement("span");
    name.className = "who";
    name.textContent = who === "user" ? "You" : "Assistant";
    const line = document.createElement("span");
    line.className = "line";
    line.textContent = who === "user" ? `"${text}"` : text;
    if (who === "assistant") line.lang = language;
    item.append(name, line);
    transcript.append(item);
    transcript.scrollTop = transcript.scrollHeight;
    $("live-transcript-empty").hidden = true;
    if (who === "assistant") {
      $("live-caption").textContent = text;
      $("live-caption").lang = language;
      $("live-caption").classList.add("live");
    }
  }

  // Called by the assistant for every sentence it says.
  function say(text, language, urgent) {
    addLine("assistant", text, language);
    // Without a voice the line still takes time to read; the assistant
    // holds back further guidance until then.
    state.busyUntil = performance.now() + Math.min(4000, 900 + text.length * 45);
    if (!$("live-sound").checked || !("speechSynthesis" in window)) return;
    const voice = shared.voiceFor(language);
    if (!voice) {
      if (state.voiceMissing !== language) {
        state.voiceMissing = language;
        setStatus(`This browser has no ${shared.languageNames[language]} voice, so the answers are shown but not spoken.`);
      }
      return;
    }
    if (urgent) speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = voice;
    utterance.lang = voice.lang;
    speechSynthesis.speak(utterance);
  }

  const isBusy = () => performance.now() < state.busyUntil
    || ("speechSynthesis" in window && (speechSynthesis.speaking || speechSynthesis.pending));

  // ------------------------------------------------------------------ models

  async function download(file, progress) {
    const response = await fetch(MODELS + file, { cache: "no-cache" });
    if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      size += value.length;
      progress(value.length);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    return bytes;
  }

  function workspace(size) {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    return { size, context: canvas.getContext("2d", { willReadFrequently: true }) };
  }

  async function open(file, progress) {
    return window.ort.InferenceSession.create(await download(file, progress),
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }

  // Loads the models and starts the assistant. Done once, on the first start.
  async function prepare() {
    setStatus("Loading the models…");
    await shared.loadScript(`${ORT}ort.wasm.min.js`);
    window.ort.env.wasm.wasmPaths = ORT;
    window.ort.env.wasm.numThreads = 1;
    window.ort.env.wasm.proxy = true; // the models run in a worker, so the page stays responsive

    const manifest = await (await fetch(`${MODELS}models.json`, { cache: "no-cache" })).json();
    let megabytes = 0;
    const progress = (bytes) => {
      megabytes += bytes / 1e6;
      setStatus(`Loading the models… ${megabytes.toFixed(0)} MB`);
    };
    const specs = [...manifest.detectors, manifest.currency];
    const sessions = await Promise.all(specs.map((spec) => open(spec.file, progress)));
    state.iou = manifest.iou;
    state.detectors = manifest.detectors.map((spec, index) => (
      { ...spec, session: sessions[index], ...workspace(spec.size) }));
    state.currency = { ...manifest.currency, session: sessions[sessions.length - 1],
      ...workspace(manifest.currency.size) };

    setStatus("Starting the assistant…");
    const pyodide = await shared.python();
    const labels = manifest.detectors.flatMap((spec) => (
      spec.keep ? spec.keep.map((index) => spec.names[index]) : spec.names));
    const create = pyodide.runPython("from visual_assist.live import LiveAssistant\nLiveAssistant");
    state.live = create(pyodide.toPy(labels), say, isBusy, $("live-language").value);
  }

  // Red, green and blue planes of a square drawing, each value 0 to 1.
  function planes({ size, context }) {
    const pixels = context.getImageData(0, 0, size, size).data;
    const area = size * size;
    const values = new Float32Array(3 * area);
    for (let i = 0, p = 0; i < area; i += 1, p += 4) {
      values[i] = pixels[p] / 255;
      values[i + area] = pixels[p + 1] / 255;
      values[i + 2 * area] = pixels[p + 2] / 255;
    }
    return new window.ort.Tensor("float32", values, [1, 3, size, size]);
  }

  async function run(spec) {
    const { session } = spec;
    const output = await session.run({ [session.inputNames[0]]: planes(spec) });
    return output[session.outputNames[0]];
  }

  function overlap(a, b) {
    const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
    const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    if (w <= 0 || h <= 0) return 0;
    const common = w * h;
    return common / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - common);
  }

  // One detector on the current picture: [{ label, confidence, box }], box in picture pixels.
  async function detect(spec) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    const { size, context } = spec;
    // Fit the picture into the model's square, grey bars filling the rest.
    const scale = Math.min(size / width, size / height);
    const padX = (size - width * scale) / 2;
    const padY = (size - height * scale) / 2;
    context.fillStyle = "rgb(114, 114, 114)";
    context.fillRect(0, 0, size, size);
    context.drawImage(video, padX, padY, width * scale, height * scale);

    const { data, dims } = await run(spec); // [1, 4 + classes, boxes]
    const classes = dims[1] - 4;
    const boxes = dims[2];
    const keep = spec.keep ? new Set(spec.keep) : null;
    const found = [];
    for (let i = 0; i < boxes; i += 1) {
      let best = 0;
      let score = 0;
      for (let c = 0; c < classes; c += 1) {
        const value = data[(4 + c) * boxes + i];
        if (value > score) {
          score = value;
          best = c;
        }
      }
      if (score < spec.confidence || (keep && !keep.has(best))) continue;
      const cx = data[i];
      const cy = data[boxes + i];
      const w = data[2 * boxes + i];
      const h = data[3 * boxes + i];
      const clamp = (value, most) => Math.max(0, Math.min(most, value));
      found.push({
        label: spec.names[best],
        kind: `${spec.file}:${best}`,
        confidence: score,
        box: [clamp((cx - w / 2 - padX) / scale, width), clamp((cy - h / 2 - padY) / scale, height),
          clamp((cx + w / 2 - padX) / scale, width), clamp((cy + h / 2 - padY) / scale, height)],
      });
    }
    // Keep the surest of any boxes of one kind that cover the same thing.
    found.sort((a, b) => b.confidence - a.confidence);
    const kept = [];
    for (const candidate of found) {
      if (kept.every((other) => other.kind !== candidate.kind
          || overlap(other.box, candidate.box) <= state.iou)) kept.push(candidate);
    }
    return kept;
  }

  // The banknote model's answer, averaged over a few pictures: { class: probability }.
  async function readNote() {
    const spec = state.currency;
    const { size, context } = spec;
    const totals = new Array(spec.names.length).fill(0);
    for (let n = 0; n < NOTE_PICTURES; n += 1) {
      // The middle square of the picture, as the model was trained.
      const scale = size / Math.min(video.videoWidth, video.videoHeight);
      context.drawImage(video, (size - video.videoWidth * scale) / 2,
        (size - video.videoHeight * scale) / 2, video.videoWidth * scale, video.videoHeight * scale);
      const { data } = await run(spec);
      data.forEach((value, index) => { totals[index] += value / NOTE_PICTURES; });
      await sleep(120);
    }
    return Object.fromEntries(spec.names.map((name, index) => [name, totals[index]]));
  }

  const thumbnailContext = (() => {
    const canvas = document.createElement("canvas");
    canvas.width = THUMBNAIL.width;
    canvas.height = THUMBNAIL.height;
    return canvas.getContext("2d", { willReadFrequently: true });
  })();

  function thumbnail() {
    thumbnailContext.drawImage(video, 0, 0, THUMBNAIL.width, THUMBNAIL.height);
    const pixels = thumbnailContext.getImageData(0, 0, THUMBNAIL.width, THUMBNAIL.height).data;
    const grey = [];
    for (let p = 0; p < pixels.length; p += 4) grey.push(Math.round((pixels[p] + pixels[p + 1] + pixels[p + 2]) / 3));
    return grey;
  }

  // ------------------------------------------------------------------- drawing

  function outline(detections) {
    const style = getComputedStyle(document.documentElement);
    const colour = (name) => style.getPropertyValue(name).trim();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    const context = overlay.getContext("2d");
    const line = Math.max(2, overlay.width / 220);
    context.lineWidth = line;
    context.font = `700 ${Math.max(13, overlay.width / 36)}px "Atkinson Hyperlegible", system-ui, sans-serif`;
    context.textBaseline = "top";
    for (const { label, confidence, box } of detections) {
      const [x1, y1, x2, y2] = box;
      const door = label === "door";
      context.strokeStyle = door ? colour("--accent") : colour("--path");
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const text = `${label} ${Math.round(confidence * 100)}%`;
      const width = context.measureText(text).width + 4 * line;
      const height = Math.max(13, overlay.width / 36) + 3 * line;
      const top = y1 >= height ? y1 - height : y1;
      context.fillStyle = door ? colour("--accent") : colour("--path");
      context.fillRect(x1 - line / 2, top, width, height);
      context.fillStyle = door ? colour("--accent-ink") : "#ffffff";
      context.fillText(text, x1 + line * 1.5, top + line * 1.5);
    }
  }

  // --------------------------------------------------------------- the camera

  async function watch(run) {
    while (state.running === run) {
      if (video.readyState < 2 || !video.videoWidth) {
        await sleep(100);
        continue;
      }
      const started = performance.now();
      try {
        // Questions first, so each is answered from the picture taken after it.
        for (const text of state.pending.splice(0)) {
          if (state.live.intent(text) === "money") state.live.note(JSON.stringify(await readNote()));
          state.live.hear(text);
        }
        const detections = [];
        for (const spec of state.detectors) detections.push(...await detect(spec));
        if (state.running !== run) return;
        state.live.see(JSON.stringify({
          w: video.videoWidth,
          h: video.videoHeight,
          thumb: thumbnail(),
          detections: detections.map(({ label, confidence, box }) => [label, confidence, ...box]),
        }));
        outline(detections);
        const seconds = (performance.now() - started) / 1000;
        const guiding = state.live.guiding;
        $("live-meter").textContent = `${detections.length} recognised · ${seconds.toFixed(1)} s per picture`
          + (guiding ? ` · guiding to the ${guiding}` : "");
      } catch (error) {
        console.error(error);
        stop();
        setStatus(`The demo stopped with an error: ${error.message || error}`);
        return;
      }
      await sleep(30);
    }
  }

  function explain(error) {
    if (error && error.name === "NotAllowedError") {
      return "The camera was not allowed. Allow it in the browser's address bar, then start again.";
    }
    if (error && (error.name === "NotFoundError" || error.name === "OverconstrainedError")) {
      return "No camera was found on this device.";
    }
    return `Could not start: ${(error && error.message) || error}`;
  }

  async function start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("This browser cannot use a camera on this page.");
      return;
    }
    toggle.disabled = true;
    const run = Date.now();
    try {
      setStatus("Waiting for permission to use the camera…");
      state.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } },
      });
      video.srcObject = state.stream;
      await video.play();
      $("camera-off").hidden = true;
      video.parentElement.classList.toggle("portrait", video.videoHeight > video.videoWidth);
      state.loading = state.loading || prepare();
      await state.loading;
    } catch (error) {
      state.loading = state.live ? state.loading : null; // allow another try
      stop();
      setStatus(explain(error));
      toggle.disabled = false;
      return;
    }
    state.running = run;
    state.pending = [];
    toggle.textContent = "Stop the camera";
    toggle.dataset.state = "playing";
    toggle.disabled = false;
    $("asks").disabled = false;
    setStatus("Ask a question, or point the camera somewhere else.");
    state.live.set_language($("live-language").value);
    state.live.greet();
    watch(run);
  }

  function stop() {
    state.running = 0;
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    video.srcObject = null;
    if (state.live) state.live.camera_off();
    if ("speechSynthesis" in window) speechSynthesis.cancel();
    overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
    $("camera-off").hidden = false;
    $("asks").disabled = true;
    $("live-meter").textContent = "";
    toggle.textContent = "Start the camera";
    toggle.dataset.state = "idle";
  }

  function ask(text) {
    text = text.trim();
    if (!text || !state.running) return;
    addLine("user", text);
    state.pending.push(text);
  }

  // ------------------------------------------------------------------ wiring

  toggle.addEventListener("click", () => {
    if (state.running) {
      stop();
      setStatus("The camera is off.");
    } else {
      start();
    }
  });
  for (const button of document.querySelectorAll("[data-ask]")) {
    button.addEventListener("click", () => ask(button.dataset.ask));
  }
  $("ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    ask($("ask").value);
    $("ask").value = "";
  });
  $("live-language").addEventListener("change", () => {
    state.voiceMissing = "";
    if (state.live) state.live.set_language($("live-language").value);
  });

  // Spoken questions, where the browser can recognise speech.
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (Recognition) {
    const mic = $("mic");
    const recognition = new Recognition();
    recognition.lang = "en-IN";
    recognition.interimResults = false;
    const listening = (on) => {
      mic.dataset.listening = String(on);
      mic.textContent = on ? "Listening…" : "Speak";
    };
    recognition.onresult = (event) => ask(event.results[0][0].transcript);
    recognition.onend = () => listening(false);
    recognition.onerror = (event) => {
      listening(false);
      if (event.error === "not-allowed") setStatus("The microphone was not allowed; type the question instead.");
    };
    mic.hidden = false;
    mic.addEventListener("click", () => {
      if (mic.dataset.listening === "true") {
        recognition.stop();
        return;
      }
      if ("speechSynthesis" in window) speechSynthesis.cancel(); // don't listen to ourselves
      try {
        recognition.start();
        listening(true);
      } catch (error) {
        listening(false);
      }
    });
  }

  window.addEventListener("pagehide", stop);
})();

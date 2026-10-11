// The demo: a simulated room drawn on a canvas and driven by the project's
// own guidance code. The preset rooms play from a recording made by that
// code (demo/recordings.json). Any other room, or any other language, is
// planned on the spot by the same Python files, run in the browser by Pyodide.
(() => {
  "use strict";

  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  // The project files the guidance needs. Nothing else is loaded: no camera,
  // speech or model code. tests/test_demo.py checks this list is enough.
  const FILES = [
    "__init__.py", "config.py", "navigator.py", "planner.py", "phrases.py",
    "adaptation.py", "commands.py", "detector.py", "assistant.py", "simulation.py",
    "lidar.py", "servo.py", "currency.py", "faces.py", "live.py",
    "locales/en.json", "locales/hi.json", "locales/ml.json", "locales/ta.json",
  ];
  const VIEW = { left: -3.6, right: 3.6, bottom: -0.7, top: 6.5 }; // metres, a square
  const HFOV = 62.2; // the camera's field of view, degrees
  const LIDAR_RANGE = 8;
  const LANGUAGE_NAMES = { en: "English", ml: "Malayalam", hi: "Hindi", ta: "Tamil" };

  const $ = (id) => document.getElementById(id);
  const canvas = $("room");
  const ctx = canvas.getContext("2d");
  const transcript = $("transcript");
  const guideButton = $("guide");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(pointer: fine)").matches;

  const state = {
    presets: {}, // name -> { things, blocks, run }
    preset: null, // name of the preset shown, or null once the room is changed
    room: { things: {}, blocks: [] },
    pose: { x: 0, y: 0, heading: 0, pan: 0 },
    trail: [[0, 0]],
    hits: [], // LiDAR readings so far, as points in the room
    rays: [], // readings still being shown as beams: { x, y, until }
    flash: 0, // time until which the camera's view is shown
    route: null,
    selected: null, // { kind: "thing", key } or { kind: "block", index }
    run: 0, // raised to cancel whatever is playing
    playing: false,
    python: null, // the guidance function, once loaded
    engine: "idle", // idle | loading | ready | failed
    waiting: false, // a run was asked for before the code was ready
  };

  // ------------------------------------------------------------------ helpers

  const clone = (value) => JSON.parse(JSON.stringify(value));
  const radians = (degrees) => (degrees * Math.PI) / 180;
  const speed = () => (reducedMotion ? 40 : Number($("speed").value));
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms / speed()));
  const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

  function tween(ms, step) {
    return new Promise((resolve) => {
      const started = performance.now();
      const total = Math.max(1, ms / speed());
      const frame = (now) => {
        const t = Math.min(1, (now - started) / total);
        step(t);
        draw();
        if (t < 1) requestAnimationFrame(frame);
        else resolve();
      };
      requestAnimationFrame(frame);
    });
  }

  function setStatus(text) {
    $("status").textContent = text;
  }

  function setEngine(engine) {
    state.engine = engine;
    $("engine").textContent = {
      idle: "Guidance code: not loaded yet.",
      loading: "Guidance code: loading the Python runtime (about 10 MB, first visit only)…",
      ready: "Guidance code: the project's own Python files, running in this page.",
      failed: "Guidance code: could not be loaded here, so only the recorded rooms play.",
    }[engine];
  }

  // ------------------------------------------------------------------ drawing

  let colours = {};
  function readColours() {
    const style = getComputedStyle(document.documentElement);
    for (const name of ["surface", "ink", "muted", "line", "accent", "accent-ink",
      "path", "lidar", "obstacle"]) {
      colours[name] = style.getPropertyValue(`--${name}`).trim();
    }
  }

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.round(canvas.clientWidth * ratio);
    const height = Math.round(canvas.clientHeight * ratio);
    if (width && (canvas.width !== width || canvas.height !== height)) {
      canvas.width = width;
      canvas.height = height;
    }
    draw();
  }

  function draw() {
    const w = canvas.width;
    const h = canvas.height;
    if (!w) return;
    const unit = w / (VIEW.right - VIEW.left); // pixels per metre
    const px = unit / (canvas.clientWidth / (VIEW.right - VIEW.left)); // one CSS pixel
    const X = (x) => (x - VIEW.left) * unit;
    const Y = (y) => (VIEW.top - y) * unit;
    const now = performance.now();
    const { pose } = state;

    ctx.clearRect(0, 0, w, h);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Floor: a one-metre grid.
    ctx.strokeStyle = colours.line;
    ctx.lineWidth = px;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    for (let x = Math.ceil(VIEW.left); x <= VIEW.right; x += 1) {
      ctx.moveTo(X(x), 0);
      ctx.lineTo(X(x), h);
    }
    for (let y = Math.ceil(VIEW.bottom); y <= VIEW.top; y += 1) {
      ctx.moveTo(0, Y(y));
      ctx.lineTo(w, Y(y));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Scale bar.
    ctx.strokeStyle = colours.muted;
    ctx.fillStyle = colours.muted;
    ctx.lineWidth = 2 * px;
    ctx.beginPath();
    ctx.moveTo(X(VIEW.left + 0.3), Y(VIEW.bottom + 0.3));
    ctx.lineTo(X(VIEW.left + 1.3), Y(VIEW.bottom + 0.3));
    ctx.stroke();
    ctx.font = `${12 * px}px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("1 m", X(VIEW.left + 0.3), Y(VIEW.bottom + 0.3) - 5 * px);

    // What the camera sees, while it takes a picture.
    if (state.flash > now) {
      const middle = radians(pose.heading + pose.pan - 90);
      ctx.fillStyle = colours.accent;
      ctx.globalAlpha = 0.22;
      ctx.beginPath();
      ctx.moveTo(X(pose.x), Y(pose.y));
      ctx.arc(X(pose.x), Y(pose.y), 3.2 * unit, middle - radians(HFOV / 2), middle + radians(HFOV / 2));
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Planned route.
    if (state.route) {
      ctx.strokeStyle = colours.ink;
      ctx.lineWidth = 2 * px;
      ctx.setLineDash([7 * px, 6 * px]);
      ctx.beginPath();
      state.route.forEach(([x, y], index) => (index ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Path walked.
    ctx.strokeStyle = colours.path;
    ctx.lineWidth = 4 * px;
    ctx.beginPath();
    state.trail.forEach(([x, y], index) => (index ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.stroke();

    // Obstacles and the door.
    const disc = ([x, y, r], fill, label, selected) => {
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(X(x), Y(y), r * unit, 0, Math.PI * 2);
      ctx.fill();
      if (selected) {
        ctx.strokeStyle = colours.ink;
        ctx.lineWidth = 2 * px;
        ctx.setLineDash([4 * px, 4 * px]);
        ctx.beginPath();
        ctx.arc(X(x), Y(y), r * unit + 5 * px, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (label) {
        ctx.fillStyle = colours.ink;
        ctx.font = `700 ${13 * px}px "Atkinson Hyperlegible", system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(label, X(x), Y(y) + r * unit + 6 * px);
      }
    };
    state.room.blocks.forEach((block, index) => disc(block, colours.obstacle, "",
      state.selected && state.selected.kind === "block" && state.selected.index === index));
    for (const [key, thing] of Object.entries(state.room.things)) {
      disc(thing, key === "door" ? colours.accent : colours.obstacle, key,
        state.selected && state.selected.kind === "thing" && state.selected.key === key);
    }

    // LiDAR: every reading so far as a dot, the latest ones as beams.
    ctx.fillStyle = colours.lidar;
    for (const [x, y] of state.hits) {
      ctx.beginPath();
      ctx.arc(X(x), Y(y), 2.2 * px, 0, Math.PI * 2);
      ctx.fill();
    }
    state.rays = state.rays.filter((ray) => ray.until > now);
    ctx.strokeStyle = colours.lidar;
    ctx.lineWidth = 1.5 * px;
    for (const ray of state.rays) {
      ctx.globalAlpha = Math.min(1, (ray.until - now) / 120) * 0.8;
      ctx.beginPath();
      ctx.moveTo(X(pose.x), Y(pose.y));
      ctx.lineTo(X(ray.x), Y(ray.y));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // The user: a body, a nose showing which way they face, and the head.
    const facing = radians(pose.heading - 90);
    const body = 0.22 * unit;
    ctx.fillStyle = colours.ink;
    ctx.beginPath();
    ctx.arc(X(pose.x), Y(pose.y), body, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(X(pose.x) + Math.cos(facing) * body * 1.75, Y(pose.y) + Math.sin(facing) * body * 1.75);
    ctx.lineTo(X(pose.x) + Math.cos(facing + 1.9) * body, Y(pose.y) + Math.sin(facing + 1.9) * body);
    ctx.lineTo(X(pose.x) + Math.cos(facing - 1.9) * body, Y(pose.y) + Math.sin(facing - 1.9) * body);
    ctx.closePath();
    ctx.fill();
    const looking = radians(pose.heading + pose.pan - 90);
    ctx.fillStyle = colours.accent;
    ctx.beginPath();
    ctx.arc(X(pose.x) + Math.cos(looking) * body * 0.45, Y(pose.y) + Math.sin(looking) * body * 0.45,
      body * 0.32, 0, Math.PI * 2);
    ctx.fill();

    if (state.rays.length || state.flash > now) requestAnimationFrame(draw);
  }

  // ------------------------------------------------------------ the transcript

  function addLine(who, text, language) {
    const item = document.createElement("li");
    item.className = who;
    const name = document.createElement("span");
    name.className = "who";
    name.textContent = who === "user" ? "User" : "Assistant";
    const line = document.createElement("span");
    line.className = "line";
    line.textContent = who === "user" ? `"${text}"` : text;
    if (who === "assistant") line.lang = language;
    item.append(name, line);
    transcript.append(item);
    transcript.scrollTop = transcript.scrollHeight;
    $("transcript-empty").hidden = true;
    if (who === "assistant") {
      $("caption").textContent = text;
      $("caption").lang = language;
      $("caption").classList.add("live");
    }
  }

  // A voice for the language, an Indian one where the browser has it.
  function voiceFor(language) {
    if (!("speechSynthesis" in window)) return null;
    const voices = speechSynthesis.getVoices();
    const code = (voice) => voice.lang.toLowerCase().replace("_", "-");
    return voices.find((voice) => code(voice) === `${language}-in`)
      || voices.find((voice) => code(voice).startsWith(language)) || null;
  }

  // Speaks a line and resolves when it has been said; null if it cannot be spoken.
  function speak(text, language) {
    if (!$("sound").checked) return null;
    const voice = voiceFor(language);
    if (!voice) return null;
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.voice = voice;
      utterance.lang = voice.lang;
      utterance.rate = Math.min(2, 0.85 + 0.25 * speed());
      utterance.onend = utterance.onerror = () => resolve();
      setTimeout(resolve, 15000); // some browsers never report the end
      speechSynthesis.speak(utterance);
    });
  }

  function explainVoice() {
    const language = $("language").value;
    if ($("sound").checked && !voiceFor(language)) {
      setStatus(`This browser has no ${LANGUAGE_NAMES[language]} voice, so the lines are shown but not spoken.`);
    }
  }

  // ------------------------------------------------------------------ playing

  function resetRun() {
    state.run += 1;
    state.playing = false;
    state.pose = { x: 0, y: 0, heading: 0, pan: 0 };
    state.trail = [[0, 0]];
    state.hits = [];
    state.rays = [];
    state.flash = 0;
    state.route = null;
    transcript.replaceChildren();
    $("transcript-empty").hidden = false;
    $("caption").textContent = "The line being spoken appears here.";
    $("caption").lang = "en";
    $("caption").classList.remove("live");
    if ("speechSynthesis" in window) speechSynthesis.cancel();
    guideButton.textContent = "Guide me to the door";
    guideButton.dataset.state = "idle";
    draw();
  }

  async function play(result, language) {
    const run = ++state.run;
    const alive = () => run === state.run;
    state.playing = true;
    guideButton.textContent = "Stop";
    guideButton.dataset.state = "playing";
    setStatus("");
    const { pose } = state;

    for (const event of result.events) {
      if (!alive()) return;
      if (event.type === "say") {
        addLine(event.who, event.text, language);
        const spoken = event.who === "assistant" ? speak(event.text, language) : null;
        if (spoken) await spoken;
        else await sleep(event.who === "user" ? 900 : Math.min(3400, 1000 + event.text.length * 38));
      } else if (event.type === "look") {
        pose.pan = event.pan;
        draw();
      } else if (event.type === "photo") {
        state.flash = performance.now() + 300 / speed();
        draw();
        await sleep(300);
      } else if (event.type === "ray") {
        const reach = event.distance === null ? LIDAR_RANGE : event.distance;
        const end = [pose.x + reach * Math.sin(radians(event.bearing)),
          pose.y + reach * Math.cos(radians(event.bearing))];
        if (event.distance !== null) state.hits.push(end);
        state.rays.push({ x: end[0], y: end[1], until: performance.now() + 160 });
        draw();
        await sleep(26);
      } else if (event.type === "route") {
        state.route = event.points;
        draw();
        await sleep(350);
      } else if (event.type === "turn") {
        const from = pose.heading;
        const by = ((event.heading - from + 540) % 360) - 180; // the short way round
        await tween(Math.abs(by) * 9 + 150, (t) => { pose.heading = from + by * t; });
      } else if (event.type === "walk") {
        const [fromX, fromY] = [pose.x, pose.y];
        const [toX, toY] = event.to;
        const metres = Math.hypot(toX - fromX, toY - fromY);
        const mark = state.trail.length;
        await tween(metres * 560, (t) => {
          pose.x = fromX + (toX - fromX) * t;
          pose.y = fromY + (toY - fromY) * t;
          state.trail[mark] = [pose.x, pose.y];
        });
      }
    }
    if (!alive()) return;

    state.playing = false;
    state.route = null;
    guideButton.textContent = "Guide me again";
    guideButton.dataset.state = "idle";
    if (result.reached) {
      setStatus(result.closest === null ? "Arrived at the door."
        : `Arrived at the door, never closer than ${result.closest} m to an obstacle.`);
    } else {
      setStatus("The assistant stopped without reaching the door. It does that when it cannot find a clear way; nobody was walked into anything.");
    }
    draw();
  }

  async function startGuide() {
    if (state.playing) {
      resetRun();
      setStatus("Stopped.");
      return;
    }
    const language = $("language").value;
    resetRun();
    const preset = state.preset && state.presets[state.preset];
    let result = null;
    if (preset && language === "en") {
      result = preset.run;
    } else if (state.python) {
      setStatus("Planning…");
      await nextFrame();
      await nextFrame();
      try {
        result = JSON.parse(state.python(JSON.stringify(state.room), language));
      } catch (error) {
        setStatus(`The guidance code stopped with an error: ${error.message}`);
        return;
      }
    } else if (state.engine === "failed") {
      setStatus("This room needs the guidance code, which could not be loaded here. The two preset rooms still play in English.");
      return;
    } else {
      state.waiting = true;
      setStatus("Loading the guidance code. The run starts by itself when it is ready…");
      return;
    }
    explainVoice();
    await play(result, language);
  }

  // ------------------------------------------------------------- the Python

  function loadScript(source) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = source;
      script.onload = resolve;
      script.onerror = () => reject(new Error(`could not load ${source}`));
      document.head.append(script);
    });
  }

  // Loads Python and the project's files once; both demos on the page share it.
  let pythonLoading = null;
  function python() {
    if (!pythonLoading) pythonLoading = loadPython();
    return pythonLoading;
  }

  async function loadPython() {
    setEngine("loading");
    try {
      await loadScript(`${PYODIDE}pyodide.js`);
      const pyodide = await window.loadPyodide({ indexURL: PYODIDE });
      pyodide.FS.mkdirTree("/project/visual_assist/locales");
      await Promise.all(FILES.map(async (name) => {
        // "no-cache": ask the server whether the copy the browser kept is
        // still current, so an old file is never mixed with new ones.
        const response = await fetch(`visual_assist/${name}`, { cache: "no-cache" });
        if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
        pyodide.FS.writeFile(`/project/visual_assist/${name}`, new Uint8Array(await response.arrayBuffer()));
      }));
      pyodide.runPython(`
import json, sys
sys.path.insert(0, "/project")
from visual_assist import simulation

def guide(room, language):
    room = json.loads(room)
    return json.dumps(simulation.record(room["things"], room["blocks"], language))
`);
      state.python = pyodide.globals.get("guide");
      setEngine("ready");
      return pyodide;
    } catch (error) {
      console.warn("Guidance code not loaded:", error);
      setEngine("failed");
      throw error;
    } finally {
      if (state.waiting) {
        state.waiting = false;
        startGuide();
      }
    }
  }

  // The live camera demo (demo/live.js) uses the same Python and helpers.
  window.visualAssist = { python, loadScript, voiceFor, languageNames: LANGUAGE_NAMES };

  // ------------------------------------------------------------------- rooms

  function showRoom(room, preset) {
    state.room = clone(room);
    state.preset = preset;
    state.selected = null;
    resetRun();
    setStatus("");
    for (const button of document.querySelectorAll(".choice")) {
      const pressed = button.dataset.room ? button.dataset.room === preset : false;
      button.setAttribute("aria-pressed", String(pressed));
    }
    $("remove").disabled = true;
  }

  function roomChanged() {
    state.preset = null;
    for (const button of document.querySelectorAll(".choice")) button.setAttribute("aria-pressed", "false");
    resetRun();
    setStatus("");
    $("remove").disabled = !state.selected || state.selected.key === "door";
  }

  const allDiscs = (room) => [...Object.values(room.things), ...room.blocks];

  // True if a disc is clear of the user's starting point and of everything else.
  function fits([x, y, r], others) {
    if (Math.hypot(x, y) - r < 0.8) return false;
    return others.every(([ox, oy, or]) => Math.hypot(x - ox, y - oy) > r + or + 0.15);
  }

  function place(others, far, radius) {
    const between = (low, high) => low + Math.random() * (high - low);
    for (let attempt = 0; attempt < 60; attempt += 1) {
      const disc = [between(-2.6, 2.6), between(1.0, far), radius];
      if (fits(disc, others)) return disc.map((value) => Math.round(value * 100) / 100);
    }
    return null;
  }

  function randomRoom() {
    const between = (low, high) => low + Math.random() * (high - low);
    const bearing = radians(between(-38, 38));
    const distance = between(3.6, 5.4) + 0.45;
    const door = [distance * Math.sin(bearing), distance * Math.cos(bearing), 0.45]
      .map((value) => Math.round(value * 100) / 100);
    const room = { things: { door }, blocks: [] };
    // Keep the doorway itself clear: treat the door as larger while placing.
    const keepClear = [door[0], door[1], 1.2];
    const chair = place([keepClear], door[1] - 0.6, 0.25);
    if (chair) room.things.chair = chair;
    const count = 2 + Math.floor(Math.random() * 3);
    for (let n = 0; n < count; n += 1) {
      const block = place([keepClear, ...allDiscs(room)], door[1] - 0.4, between(0.2, 0.36));
      if (block) room.blocks.push(block);
    }
    return room;
  }

  // ---------------------------------------------------------------- dragging

  function toRoom(event) {
    const box = canvas.getBoundingClientRect();
    const unit = box.width / (VIEW.right - VIEW.left);
    return [VIEW.left + (event.clientX - box.left) / unit, VIEW.top - (event.clientY - box.top) / unit];
  }

  function discAt([x, y]) {
    for (const [key, [cx, cy, r]] of Object.entries(state.room.things)) {
      if (Math.hypot(x - cx, y - cy) <= r + 0.12) return { kind: "thing", key };
    }
    for (let index = state.room.blocks.length - 1; index >= 0; index -= 1) {
      const [cx, cy, r] = state.room.blocks[index];
      if (Math.hypot(x - cx, y - cy) <= r + 0.12) return { kind: "block", index };
    }
    return null;
  }

  const discOf = (pick) => (pick.kind === "thing" ? state.room.things[pick.key] : state.room.blocks[pick.index]);

  function moveDisc(pick, x, y) {
    const disc = discOf(pick);
    const r = disc[2];
    x = Math.max(VIEW.left + r + 0.1, Math.min(VIEW.right - r - 0.1, x));
    y = Math.max(VIEW.bottom + r + 0.1, Math.min(VIEW.top - r - 0.3, y));
    // Nothing may be put on top of the user.
    const fromUser = Math.hypot(x, y);
    const least = r + 0.85;
    if (fromUser < least) {
      const scale = least / Math.max(fromUser, 0.01);
      x *= scale;
      y = Math.max(y * scale, 0.2);
    }
    disc[0] = Math.round(x * 100) / 100;
    disc[1] = Math.round(y * 100) / 100;
  }

  let drag = null;
  canvas.addEventListener("pointerdown", (event) => {
    const point = toRoom(event);
    const pick = discAt(point);
    state.selected = pick;
    $("remove").disabled = !pick || pick.key === "door";
    if (pick && finePointer) {
      const disc = discOf(pick);
      drag = { pick, dx: disc[0] - point[0], dy: disc[1] - point[1], moved: false };
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("grabbing");
    }
    draw();
  });
  canvas.addEventListener("pointermove", (event) => {
    const point = toRoom(event);
    if (!drag) {
      canvas.classList.toggle("can-grab", finePointer && Boolean(discAt(point)));
      return;
    }
    if (!drag.moved) {
      drag.moved = true;
      roomChanged();
    }
    moveDisc(drag.pick, point[0] + drag.dx, point[1] + drag.dy);
    draw();
  });
  const endDrag = () => {
    drag = null;
    canvas.classList.remove("grabbing");
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  // ------------------------------------------------------------------ wiring

  for (const button of document.querySelectorAll(".choice[data-room]")) {
    button.addEventListener("click", () => {
      const preset = state.presets[button.dataset.room];
      if (preset) showRoom(preset, button.dataset.room);
    });
  }
  $("random").addEventListener("click", () => {
    showRoom(randomRoom(), null);
    $("random").setAttribute("aria-pressed", "true");
  });
  $("add").addEventListener("click", () => {
    const door = state.room.things.door;
    const block = place(allDiscs(state.room), door ? door[1] - 0.3 : 4.5, 0.3);
    if (!block) {
      setStatus("There is no room left for another obstacle.");
      return;
    }
    state.room.blocks.push(block);
    state.selected = { kind: "block", index: state.room.blocks.length - 1 };
    roomChanged();
  });
  $("remove").addEventListener("click", () => {
    const pick = state.selected;
    if (!pick || pick.key === "door") return;
    if (pick.kind === "block") state.room.blocks.splice(pick.index, 1);
    else delete state.room.things[pick.key];
    state.selected = null;
    roomChanged();
  });
  guideButton.addEventListener("click", startGuide);
  $("language").addEventListener("change", () => {
    if (!state.playing) resetRun();
    setStatus("");
    explainVoice();
  });
  $("sound").addEventListener("change", explainVoice);

  if (!finePointer) $("drag-hint").textContent = "Change the room with the buttons, then ask.";

  const repository = $("repo");
  if (location.hostname.endsWith(".github.io")) {
    const owner = location.hostname.split(".")[0];
    const name = location.pathname.split("/").filter(Boolean)[0];
    if (name) repository.href = `https://github.com/${owner}/${name}`;
  }

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    readColours();
    draw();
  });
  new ResizeObserver(resize).observe(canvas);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);

  readColours();
  setEngine("idle");
  fetch("demo/recordings.json", { cache: "no-cache" })
    .then((response) => (response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`))))
    .then((presets) => {
      state.presets = presets;
      showRoom(presets.chair, "chair");
    })
    .catch((error) => {
      console.warn("Recorded rooms not loaded:", error);
      showRoom(randomRoom(), null);
    })
    .finally(() => {
      resize();
      // Fetch the Python runtime once the page has settled; it is only
      // needed for rooms and languages that were not recorded.
      setTimeout(() => python().catch(() => {}), 600);
    });
})();

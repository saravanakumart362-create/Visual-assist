// "Inside the device": one spoken command followed from part to part.
// Two commands come from demo/device.json, recorded by the project's own
// LiDAR and servo drivers talking to simulated parts (see trace() in
// visual_assist/simulation.py). The third, "Your camera, live", is whatever
// the visitor last asked in the camera section: demo/live.js reports the
// steps that question really went through. This file only shows steps: it
// lights the parts at work on the diagram, moves a dot along the cables in
// use, and lists what was sent.
(() => {
  "use strict";

  const SVG = "http://www.w3.org/2000/svg";
  const HFOV = 62.2; // the camera's field of view, degrees
  const SWEEP_RANGE = 6; // metres the route planner looks ahead
  const DWELL_MS = 3500; // shortest time Play stays on a step

  const $ = (id) => document.getElementById(id);
  const diagram = $("diagram");
  const packet = $("packet");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // What each kind of step lights up, and the cables its data travels, in
  // order. A cable ending in "<" is travelled backwards.
  const PARTS = {
    hear: { nodes: ["wearer", "mic", "stt"], hops: ["voice", "usbaudio"] },
    understand: { nodes: ["stt", "cmd", "hub"], hops: ["text", "cmd"] },
    speak: { nodes: ["hub", "phr", "tts", "ear", "wearer"], hops: ["say", "tts", "audio", "sound"] },
    turn: { nodes: ["hub", "head", "esp", "servo"], hops: ["head", "usb", "pwm"] },
    see: { nodes: ["cam", "det", "hub"], hops: ["csi", "det"] },
    measure: { nodes: ["lidar", "lid", "hub"], hops: ["uart", "lid"] },
    look: {
      nodes: ["hub", "head", "esp", "servo", "cam", "det", "lidar", "lid"],
      hops: ["head", "usb", "pwm", "csi", "det", "uart", "lid"],
    },
    aim: {
      nodes: ["hub", "head", "esp", "servo", "lidar", "lid"],
      hops: ["head", "usb", "pwm", "uart", "lid"],
    },
    plan: { nodes: ["hub", "plan"], hops: ["plan", "plan<"] },
  };
  PARTS.centre = PARTS.turn;
  PARTS.sweep = PARTS.aim;
  // A question put to the visitor's own camera. Some of these depend on how
  // it was asked and answered, so they are worked out from the step.
  const NOTHING = { nodes: [], hops: [] };
  Object.assign(PARTS, {
    "live-wait": NOTHING,
    "live-hear": (step) => (step.spoken ? PARTS.hear : { nodes: ["wearer"], hops: [] }),
    "live-see": PARTS.see,
    "live-note": PARTS.see,
    "live-faces": PARTS.see,
    "live-absent": { ...NOTHING, missing: ["head", "esp", "servo", "lidar", "lid"] },
    "live-speak": (step) => (step.aloud ? PARTS.speak : { nodes: ["hub", "phr"], hops: ["say"] }),
  });
  const understood = PARTS.understand;
  PARTS.understand = (step) => (step.typed ? { nodes: ["cmd", "hub"], hops: ["cmd"] } : understood);
  const partsOf = (step) => (typeof PARTS[step.kind] === "function" ? PARTS[step.kind](step) : PARTS[step.kind]);

  // What a phone or laptop does not have. Shown faded while the camera is followed.
  const NOT_HERE = { nodes: ["head", "esp", "servo", "lidar", "lid", "plan"], wires: ["head", "usb", "pwm", "uart", "lid", "plan"] };
  const CAPTION = {
    device: $("diagram-caption").textContent,
    live: "Your camera stands in for the Pi Camera, your microphone and speakers for the USB microphone and the earphone, and this page for the Raspberry Pi. The faded parts are not on a phone or laptop.",
  };

  const state = {
    scenes: {}, // name -> { command, steps }; "live" appears once a question is answered
    scene: "find",
    index: 0,
    playing: null, // the timer that moves on to the next step
    motion: 0, // raised to stop the dot that is travelling
    camera: false, // whether the visitor's camera is running
  };

  // ------------------------------------------------------------------ wording

  const metres = (value, places = 2) => `${value.toFixed(places)} m`;
  const about = (value) => `about ${Number(value.toFixed(1))} m`;
  const quoted = (text) => `“${text}”`;

  function where(bearing) {
    if (bearing === 0) return "straight ahead";
    return `${Math.abs(bearing)}° to the ${bearing < 0 ? "left" : "right"}`;
  }

  const PLACES = { left: "on the left of the picture", center: "in the middle of the picture", right: "on the right of the picture" };

  function seen(objects) {
    if (!objects.length) return [["Detection", "nothing it knows"]];
    return objects.map((object) => ["Detection",
      `${object.label}, ${Math.round(object.confidence * 100)}% sure, ${PLACES[object.position]}`
      // Some things come in every size, so theirs says nothing of the distance.
      + (object.looks === null ? "" : `, looks ${about(object.looks)} away`)]);
  }

  // The banknote model's classes are named like "500", "100_new" and "none".
  function rupees(name) {
    if (name === "none") return "no note";
    const [value, series] = name.split("_");
    return series ? `₹${value}, ${series} series` : `₹${value}`;
  }

  const reading = (value) => (value === null ? "no reading: nothing in range" : metres(value));

  function route(legs) {
    return legs.map(([bearing, length], index) => {
      const turn = bearing === 0 ? "straight on" : `${Math.abs(bearing)}° ${bearing < 0 ? "left" : "right"}`;
      return `${index ? "then " : ""}${turn} for ${metres(length, 1)}`;
    }).join(", ");
  }

  // Each kind of step in words: a heading, a paragraph, and what was sent
  // as [who, what] pairs.
  const TELL = {
    hear: (step) => ({
      heading: "You speak",
      body: "A USB microphone turns the voice into numbers, 16,000 of them a second, and sends them to the Raspberry Pi. A speech recogniser there turns the sound into words.",
      sent: [["Microphone → Pi", "sound, 16,000 samples a second"], ["Speech to text", quoted(step.text), "said"]],
    }),
    understand: (step) => ({
      heading: "The words become a command",
      body: "commands.py compares the words with the commands it knows and with the names of the things the detector can recognise. "
        + (step.intent ? "The assistant now knows what to do and what to do it with."
          : "These words matched none of them. The assistant then stays quiet: a microphone overhears a great deal that is not meant for it."),
      sent: [["Command", step.intent ? [step.intent, step.target].filter(Boolean).join(": ") : "none"]],
    }),
    speak: (step, last, scene) => ({
      heading: last ? "The answer" : "The assistant speaks",
      body: "phrases.py builds the sentence in the wearer's language, a speech synthesiser turns it into sound, and the Pi's audio jack carries it to the earphone."
        + (last && scene === "guide" ? " Once the wearer has turned, the device looks, sweeps and plans again for the next instruction." : ""),
      sent: [["Sentence", quoted(step.text), "said"], ["Pi → earphone", "the sentence as sound"]],
    }),
    turn: (step) => ({
      heading: `The head turns ${where(step.bearing)}`,
      body: "The Raspberry Pi does not drive the servos itself. servo.py sends the ESP8266 a line of text down the USB cable. The ESP8266 changes the width of the pulse it sends that servo 50 times a second, and answers OK. A pulse of 500 millionths of a second is one end of the servo's travel, 2,500 is the other, and 1,500 (PAN 90) is straight ahead.",
      sent: [
        ["Pi → ESP8266", step.sent[0]], ["ESP8266 → pan servo", `pulses of ${step.pulses_us[0]} µs`], ["ESP8266 → Pi", step.replies[0]],
        ["Pi → ESP8266", step.sent[1]], ["ESP8266 → tilt servo", `pulses of ${step.pulses_us[1]} µs`], ["ESP8266 → Pi", step.replies[1]],
      ],
    }),
    see: (step) => ({
      heading: "The camera takes a picture",
      body: "A picture of 640 × 480 pixels comes up the ribbon cable. Two YOLOv5 models go through it, one for everyday objects and one trained on doors. They name what they find and say where in the picture it is. How large a thing looks tells roughly how far away it is.",
      sent: [["Camera → Pi", "one picture, 640 × 480"], ...seen(step.objects)],
    }),
    measure: (step) => ({
      heading: "The LiDAR measures",
      body: "The TF-Luna sends its reading as 9 bytes on a serial wire, about 100 times a second. lidar.py takes the newest, checks it against its last byte, and reads the distance from the third and fourth. The LiDAR points the same way as the camera, so it measures to whatever is in the middle of the picture.",
      sent: [["LiDAR → Pi", step.frame], ["LiDAR reader", reading(step.metres)]],
    }),
    look: (step) => ({
      heading: `Next view: ${where(step.bearing)}`,
      body: "The same three things again: turn the head, take a picture, measure along the middle of it. Five views cover the room in front of the wearer.",
      sent: [
        ["Pi → ESP8266", step.sent.join(", then ")], ["ESP8266 → Pi", step.replies.join(", ")],
        ["Camera → Pi", "one picture, 640 × 480"], ...seen(step.objects),
        ["LiDAR → Pi", step.frame], ["LiDAR reader", reading(step.metres)],
      ],
    }),
    aim: (step) => {
      let verdict;
      if (!step.approx) {
        verdict = "Here the two agree, so the LiDAR's figure, the exact one, is used.";
      } else if (step.metres === null) {
        verdict = "Here the LiDAR has no reading, so the camera's estimate is used and the answer will say \"about\".";
      } else {
        verdict = `Here they do not: the LiDAR says ${metres(step.metres, 1)} and the size of the ${step.target} says ${about(step.looks)}. `
          + "The beam has been stopped by something nearer, so the camera's estimate is used and the answer will say \"about\".";
      }
      return {
        heading: `The LiDAR is aimed at the ${step.target}`,
        body: `The pictures put the ${step.target} ${where(step.bearing)}, so the head turns there for the LiDAR to measure it. `
          + `The reading is then checked against the distance the camera judged from the ${step.target}'s size. ${verdict}`,
        sent: [
          ["Pi → ESP8266", step.sent.join(", then ")], ["ESP8266 → Pi", step.replies.join(", ")],
          ["LiDAR → Pi", step.frame], ["LiDAR reader", reading(step.metres)],
          ["Camera's estimate", about(step.looks)],
          ["Distance used", step.approx ? `${about(step.distance)}, from the camera` : `${metres(step.distance)}, from the LiDAR`],
        ],
      };
    },
    centre: (step) => ({
      heading: "The head faces forward again",
      body: "Between jobs the head points straight ahead, so the LiDAR can warn of whatever the wearer is walking towards.",
      sent: [["Pi → ESP8266", step.sent.join(", then ")], ["ESP8266 → Pi", step.replies.join(", ")]],
    }),
    sweep: (step) => {
      const hits = step.readings.filter((ray) => ray.metres !== null).length;
      const first = step.readings[0].bearing;
      const last = step.readings[step.readings.length - 1].bearing;
      const pace = step.readings[1].bearing - first;
      return {
        heading: "The LiDAR sweeps the room",
        body: `To plan a route the Pi must know where things are, not only what they are. The head turns from ${where(first)} to ${where(last)} `
          + `in steps of ${pace}°, and the LiDAR measures at each one: ${step.readings.length} readings and no pictures. `
          + `${hits === 1 ? "One of them" : `${hits} of them`} met something.`,
        sent: step.readings.map((ray) => [ray.sent, `${ray.frame} = ${ray.metres === null ? "nothing" : metres(ray.metres)}`]),
      };
    },
    plan: (step) => {
      const hits = step.rays.filter(([, distance]) => distance !== null).length;
      const named = step.obstacles.map((obstacle) => obstacle[3]).filter(Boolean);
      return {
        heading: "The route is planned",
        body: "planner.py puts every LiDAR reading, and every object the camera named, on a map of 10 cm squares. It then searches for the shortest way to the target that stays 35 cm clear of them all."
          + (step.blocker ? ` The camera named the thing on the straight line a ${step.blocker}, so the wearer is told what is in the way.` : ""),
        sent: [
          ["From the sweep", `${step.rays.length} readings, ${hits} with something there`],
          ["From the camera", named.length ? named.join(", ") : "nothing but the target"],
          ["Route", step.legs.length ? route(step.legs) : "none found"],
          ...(step.direct ? [] : [["In the way", step.blocker || "something not recognised"]]),
        ],
      };
    },

    // ---- a question put to the visitor's own camera
    "live-wait": (step) => ({
      heading: step.running ? "Ask your camera a question" : "Start your camera first",
      body: step.running
        ? "Ask with the buttons below. The parts will light up in turn with what really happened to your question: the picture your camera took, what the models found in it, and the sentence that was built from that."
        : "This one is not a recording. It follows a real question through the parts, with your own camera in place of the Pi Camera.",
      sent: [],
    }),
    "live-hear": (step) => ({
      heading: step.spoken ? "You speak" : "You type the question",
      body: step.spoken
        ? "Your microphone picked up the voice and your browser's speech recogniser turned it into words. On the device a USB microphone and a recogniser on the Raspberry Pi do the same."
        : "Typed words need no microphone and no speech recogniser, so those two parts are passed by. On the device the question is spoken.",
      sent: step.spoken
        ? [["Microphone → page", "your voice"], ["Speech to text", quoted(step.text), "said"]]
        : [["Typed", quoted(step.text), "said"]],
    }),
    "live-see": (step) => ({
      heading: "Your camera takes a picture",
      body: `Your camera sent a picture of ${step.size.join(" × ")} pixels. The project's two detection models, running in this page, went through it: one for everyday objects, one trained on doors. `
        + "They name what they find and say where in the picture it is. How large a thing looks tells roughly how far away it is.",
      sent: [["Camera → page", `one picture, ${step.size.join(" × ")}`], ...seen(step.objects)],
    }),
    "live-note": (step) => ({
      heading: "The banknote model reads the note",
      body: `A banknote is not looked for by the detection models. The middle of ${step.pictures} pictures goes to a model trained on Indian notes, which gives every kind of note a score, and the scores are averaged. `
        + "Under 50% counts as no note at all; under 80% the answer is offered as a guess. On the diagram this model shares the Detection box; its code is currency.py.",
      sent: [["Camera → page", `${step.pictures} pictures`],
        ...step.scores.map(([name, score]) => ["Banknote model", `${rupees(name)}: ${Math.round(score * 100)}%`])],
    }),
    "live-faces": (step) => {
      const { faces, registered, least } = step.people;
      const each = faces.map((face, index) => {
        const which = faces.length > 1 ? `Face ${index + 1}` : "The face";
        if (!registered.length) return [which, "cannot be named: nobody is registered yet"];
        const likeness = face.likeness.toFixed(2);
        return [which, face.name
          ? `${face.name}: likeness ${likeness}, and a name needs ${least}`
          : `not known: likeness ${likeness} to the nearest registered face, under the ${least} a name needs`];
      });
      return {
        heading: "The face models look at the picture",
        body: "Two more models, running in this page as they do on the device. The first finds each face and five points on it: the eyes, the nose, the corners of the mouth. "
          + "The second takes the face, straightened by those points, and turns it into 128 numbers. Those are compared with the numbers of every registered face: the more alike two faces are, the nearer their likeness is to 1.",
        sent: [
          ["Camera → page", `one picture, ${step.size.join(" × ")}`],
          ["Face detector", faces.length ? `${faces.length} ${faces.length === 1 ? "face" : "faces"} close enough to tell apart` : "no face close enough to tell apart"],
          ...each,
          ["Registered", registered.length ? registered.join(", ") : "nobody yet"],
        ],
      };
    },
    "live-absent": (step) => {
      const thing = step.target || "nearest thing";
      const would = {
        scan: "The device would now turn its head to four more angles, taking a picture and a LiDAR reading at each. This page has no turning head and no LiDAR, so it answers from the one picture.",
        guide: `The device would now measure the distance to the ${thing} with its LiDAR, sweep the room and plan a way round whatever is in between, as "Guide me to the door" shows. `
          + "This page has no LiDAR and no turning head, so it says only which way to turn and how far the camera judges it to be.",
      };
      return {
        heading: "What the device would do next",
        body: would[step.intent] || `The device would now turn its head towards the ${thing} and measure the distance with its LiDAR, as "Where is the door?" shows. `
          + "This page has neither, so the distance is the camera's estimate and the answer says \"about\".",
        sent: [["Not on this page", "ESP8266, servos, LiDAR"]],
      };
    },
    "live-speak": (step, last) => ({
      heading: last ? "The answer" : "The assistant speaks",
      body: "phrases.py builds the sentence in the chosen language. " + (step.aloud
        ? "Your browser's voice then speaks it. On the device a speech synthesiser on the Raspberry Pi does, into the earphone."
        : "Here it is shown but not spoken: sound is switched off, or this browser has no voice for the language. On the device it is spoken into the earphone."),
      sent: [["Sentence", quoted(step.text), "said"], ...(step.aloud ? [["Page → speakers", "the sentence as sound"]] : [])],
    }),
  };

  // ------------------------------------------------------------------ diagram

  const wire = (name) => $(`w-${name}`);

  function light(parts) {
    for (const lit of diagram.querySelectorAll(".on, .missing, .absent")) lit.classList.remove("on", "missing", "absent");
    if (state.scene === "live") {
      for (const name of NOT_HERE.nodes) $(`n-${name}`).classList.add("absent");
      for (const name of NOT_HERE.wires) wire(name).classList.add("absent");
    }
    for (const name of parts.nodes) $(`n-${name}`).classList.add("on");
    for (const hop of parts.hops) wire(hop.replace("<", "")).classList.add("on");
    // The parts a step is about but that are not there to do it.
    for (const name of parts.missing || []) $(`n-${name}`).classList.replace("absent", "missing");
  }

  function hopTime(hop) {
    const length = wire(hop.replace("<", "")).querySelector("path").getTotalLength();
    return Math.max(280, length * 4.5);
  }

  const loopTime = (hops) => hops.reduce((total, hop) => total + hopTime(hop), 0);

  // Send the dot along each cable in turn, over and over, until another
  // step takes its place.
  function travel(hops) {
    const mine = ++state.motion;
    if (reducedMotion || !hops.length) {
      packet.setAttribute("hidden", "");
      return;
    }
    const legs = hops.map((hop) => {
      const path = wire(hop.replace("<", "")).querySelector("path");
      return { path, length: path.getTotalLength(), back: hop.endsWith("<"), time: hopTime(hop) };
    });
    const rest = 700; // a pause before the journey starts again
    const total = legs.reduce((sum, leg) => sum + leg.time, 0) + rest;
    const started = performance.now();

    const frame = (now) => {
      if (mine !== state.motion) return;
      let t = (now - started) % total;
      const leg = legs.find((candidate) => {
        if (t < candidate.time) return true;
        t -= candidate.time;
        return false;
      });
      if (leg) {
        const along = (t / leg.time) * leg.length;
        const point = leg.path.getPointAtLength(leg.back ? leg.length - along : along);
        packet.setAttribute("cx", point.x);
        packet.setAttribute("cy", point.y);
        packet.removeAttribute("hidden");
      } else {
        packet.setAttribute("hidden", "");
      }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------- the 9 bytes, named

  function frameTable(hex) {
    const bytes = hex.split(" ").map((pair) => parseInt(pair, 16));
    const distance = bytes[2] + bytes[3] * 256;
    const strength = bytes[4] + bytes[5] * 256;
    const sum = bytes.slice(0, 8).reduce((total, byte) => total + byte, 0) % 256;
    const pair = (index, count = 2) => hex.split(" ").slice(index, index + count).join(" ");
    const rows = [
      [pair(0), "Start of a frame: always 59 59."],
      [pair(2), `Distance, low byte first: ${distance} cm.`],
      [pair(4), strength < 100
        ? `Strength of the returning light: ${strength}. Under 100 is too weak to trust, so this frame gives no reading.`
        : `Strength of the returning light: ${strength}, enough to trust.`],
      [pair(6), "Temperature of the sensor. Not used."],
      [pair(8, 1), `Checksum: the other eight added up, keeping only the last byte of the total. ${sum === bytes[8] ? "It matches" : "It does not match"}.`],
    ];

    const table = document.createElement("table");
    table.className = "frame";
    const caption = table.createCaption();
    caption.textContent = "The 9 bytes from the LiDAR";
    const head = table.createTHead().insertRow();
    for (const title of ["Bytes", "Meaning"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = title;
      head.append(cell);
    }
    const body = table.createTBody();
    for (const [digits, meaning] of rows) {
      const row = body.insertRow();
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = digits;
      row.append(name);
      row.insertCell().textContent = meaning;
    }
    return table;
  }

  // ------------------------------------------------- the room, seen from above

  const VIEW = { width: 340, height: 176, x: 170, y: 164, scale: 24 }; // 24 units to a metre

  function shape(name, attributes, text) {
    const node = document.createElementNS(SVG, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const spot = (x, y) => [VIEW.x + x * VIEW.scale, VIEW.y - y * VIEW.scale];

  function polar(bearing, distance) {
    const angle = (bearing * Math.PI) / 180;
    return spot(distance * Math.sin(angle), distance * Math.cos(angle));
  }

  // What the device knows at this step, drawn with the wearer at the bottom.
  function above(step) {
    const svg = shape("svg", { viewBox: `0 0 ${VIEW.width} ${VIEW.height}`, role: "img" });
    const keys = new Map([["wearer", "Wearer"]]);
    const told = [];

    for (const ring of [2, 4, 6]) {
      const [x1, y1] = polar(-80, ring);
      const [x2, y2] = polar(80, ring);
      const r = ring * VIEW.scale;
      svg.append(shape("path", { class: "ring", d: `M${x1} ${y1} A${r} ${r} 0 0 1 ${x2} ${y2}` }));
      svg.append(shape("text", { x: x2 + 3, y: y2 + 3 }, `${ring} m`));
    }

    if (["turn", "see", "look", "centre", "live-see"].includes(step.kind)) {
      // The slice of the room the camera now has in its picture.
      const fov = step.hfov || HFOV;
      const [x1, y1] = polar(step.bearing - fov / 2, SWEEP_RANGE);
      const [x2, y2] = polar(step.bearing + fov / 2, SWEEP_RANGE);
      const r = SWEEP_RANGE * VIEW.scale;
      svg.append(shape("path", { class: "view", d: `M${VIEW.x} ${VIEW.y} L${x1} ${y1} A${r} ${r} 0 0 1 ${x2} ${y2} Z` }));
      keys.set("view", "In the camera's picture");
      told.push(`The camera faces ${where(step.bearing)}.`);
    }

    const beams = step.readings ? step.readings.map((ray) => [ray.bearing, ray.metres])
      : step.rays || (step.frame ? [[step.bearing, step.metres]] : []);
    for (const [bearing, distance] of beams) {
      const [x, y] = polar(bearing, distance === null ? SWEEP_RANGE : distance);
      if (step.kind !== "plan") {
        svg.append(shape("line", { class: distance === null ? "beam miss" : "beam", x1: VIEW.x, y1: VIEW.y, x2: x, y2: y }));
      }
    }
    for (const [x, y, radius, label] of step.obstacles || []) {
      if (!label) continue; // unnamed ones are the LiDAR's own readings, drawn below
      const [cx, cy] = spot(x, y);
      svg.append(shape("circle", { class: "thing", cx, cy, r: radius * VIEW.scale }));
      svg.append(shape("text", { class: "name", x: cx + radius * VIEW.scale + 4, y: cy + 3 }, label));
      keys.set("obstacle", "Named by the camera");
      told.push(`The camera put a ${label} ${metres(Math.hypot(x, y), 1)} away.`);
    }
    for (const object of step.objects || []) {
      if (object.looks === null) continue; // no telling how far: it cannot be placed
      // A real camera may judge something further off than the drawing reaches.
      const [cx, cy] = polar(object.bearing, Math.min(object.looks, SWEEP_RANGE + 0.5));
      const target = object.label === "door";
      svg.append(shape("circle", { class: target ? "goal" : "thing", cx, cy, r: 5 }));
      svg.append(shape("text", { class: "name", x: cx + 8, y: cy + 3 }, object.label));
      keys.set(target ? "door" : "obstacle", target ? "Door" : "Named by the camera");
      told.push(`A ${object.label} looks ${about(object.looks)} away, ${where(object.bearing)}.`);
    }
    const goal = step.goal || (step.kind === "aim" ? [step.bearing, step.looks] : null);
    if (goal) {
      const [cx, cy] = polar(goal[0], goal[1]);
      svg.append(shape("circle", { class: "goal", cx, cy, r: 6 }));
      svg.append(shape("text", { class: "name", x: cx + 9, y: cy + 3 }, step.target || "door"));
      keys.set("door", step.kind === "aim" ? "Where the camera puts it" : "Door");
      told.push(`The ${step.target || "door"} is ${where(goal[0])}, ${about(goal[1])} away.`);
    }
    if (step.legs && step.legs.length) {
      let [x, y] = [0, 0];
      const points = [spot(0, 0)];
      for (const [bearing, length] of step.legs) {
        const angle = (bearing * Math.PI) / 180;
        x += length * Math.sin(angle);
        y += length * Math.cos(angle);
        points.push(spot(x, y));
      }
      svg.append(shape("polyline", { class: "route", points: points.map((point) => point.join(",")).join(" ") }));
      keys.set("route", "Planned route");
      told.push(`The route goes ${route(step.legs)}.`);
    }
    const hits = beams.filter(([, distance]) => distance !== null);
    for (const [bearing, distance] of hits) {
      const [cx, cy] = polar(bearing, distance);
      svg.append(shape("circle", { class: "hit", cx, cy, r: 3.5 }));
      keys.set("lidar", "LiDAR reading");
    }
    if (beams.length) {
      told.push(hits.length
        ? `The LiDAR met something ${hits.map(([bearing, distance]) => `${metres(distance, 1)} away ${where(bearing)}`).join(", and ")}.`
        : "The LiDAR met nothing.");
    }
    svg.append(shape("circle", { class: "wearer", cx: VIEW.x, cy: VIEW.y, r: 5 }));
    svg.setAttribute("aria-label", `Seen from above, the wearer at the bottom. ${told.join(" ")}`);

    const figure = document.createElement("figure");
    figure.className = "above";
    const legend = document.createElement("ul");
    legend.className = "legend";
    for (const [key, name] of keys) {
      const item = document.createElement("li");
      const mark = document.createElement("span");
      mark.className = `key key-${key}`;
      item.append(mark, name);
      legend.append(item);
    }
    const caption = document.createElement("figcaption");
    caption.className = "sr-only";
    caption.textContent = "What the device knows at this step, seen from above.";
    figure.append(svg, legend, caption);
    return figure;
  }

  // The picture the visitor's camera took, as the models saw it.
  function shot(step) {
    const figure = document.createElement("figure");
    figure.className = "shot";
    const picture = document.createElement("img");
    picture.src = step.picture;
    picture.alt = step.objects && step.objects.length
      ? `The picture from your camera, with ${step.objects.map((object) => object.label).join(", ")} outlined.`
      : "The picture from your camera.";
    const caption = document.createElement("figcaption");
    caption.textContent = "The picture your camera sent. It has not left this page.";
    figure.append(picture, caption);
    return figure;
  }

  // --------------------------------------------------------------- the player

  // The steps being shown. Until a question has been put to the camera,
  // "Your camera, live" has one step, which says how to begin.
  function steps() {
    const scene = state.scenes[state.scene];
    return scene ? scene.steps : [{ kind: "live-wait", running: state.camera }];
  }

  function show() {
    const all = steps();
    const step = all[state.index];
    const last = state.index === all.length - 1;
    const told = TELL[step.kind](step, last, state.scene);
    const parts = partsOf(step);
    const live = state.scene === "live";

    light(parts);
    travel(parts.hops);

    $("device-heading").textContent = told.heading;
    $("device-body").textContent = told.body;
    $("device-count").textContent = all.length > 1 ? `Step ${state.index + 1} of ${all.length}` : "";
    $("device-back").disabled = state.index === 0;
    $("device-next").disabled = $("device-play").disabled = all.length === 1;
    $("device-next").textContent = last && all.length > 1 ? "Start again" : "Next";
    $("diagram-caption").textContent = live ? CAPTION.live : CAPTION.device;
    $("device-asks").hidden = !(live && state.camera);
    $("device-camera-off").hidden = !(live && !state.camera);
    $("device-sent-box").hidden = !told.sent.length;

    const list = $("device-sent");
    list.replaceChildren(...told.sent.map(([who, what, style]) => {
      const item = document.createElement("li");
      const from = document.createElement("span");
      from.className = "who";
      from.textContent = who;
      const data = document.createElement("span");
      data.className = style || "data";
      data.textContent = what;
      item.append(from, data);
      return item;
    }));
    list.classList.toggle("long", told.sent.length > 8); // the sweep: scroll, don't sprawl
    list.scrollTop = 0;

    const extra = [];
    if (step.picture) extra.push(shot(step));
    if (step.frame) extra.push(frameTable(step.frame));
    if ("bearing" in step || step.readings || step.rays) extra.push(above(step));
    $("device-extra").replaceChildren(...extra);
  }

  function go(index) {
    const count = steps().length;
    state.index = ((index % count) + count) % count;
    show();
  }

  function stop() {
    clearTimeout(state.playing);
    state.playing = null;
    $("device-play").textContent = "Play";
    $("device-play").setAttribute("aria-pressed", "false");
  }

  // Play moves on by itself, staying on each step for at least one journey
  // of the dot, and stops at the last.
  function play() {
    const all = steps();
    if (all.length === 1) return;
    if (state.index === all.length - 1) go(0);
    $("device-play").textContent = "Pause";
    $("device-play").setAttribute("aria-pressed", "true");
    const wait = () => {
      const { hops } = partsOf(all[state.index]);
      state.playing = setTimeout(() => {
        if (state.index === all.length - 1) {
          stop();
          return;
        }
        go(state.index + 1);
        wait();
      }, Math.max(DWELL_MS, loopTime(hops) + 1500));
    };
    wait();
  }

  function choose(scene) {
    stop();
    state.scene = scene;
    for (const button of document.querySelectorAll("[data-scene]")) {
      button.setAttribute("aria-pressed", String(button.dataset.scene === scene));
    }
    go(0);
  }

  function start(scenes) {
    state.scenes = scenes;
    for (const button of document.querySelectorAll("[data-scene]")) {
      button.addEventListener("click", () => choose(button.dataset.scene));
    }
    $("device-back").addEventListener("click", () => {
      stop();
      go(state.index - 1);
    });
    $("device-next").addEventListener("click", () => {
      stop();
      go(state.index + 1);
    });
    $("device-play").addEventListener("click", () => (state.playing === null ? play() : stop()));
    choose(state.scene);

    // ---- the visitor's own camera (demo/live.js)
    for (const button of document.querySelectorAll("[data-device-ask]")) {
      button.addEventListener("click", () => {
        document.dispatchEvent(new CustomEvent("visualassist:ask", { detail: { text: button.dataset.deviceAsk } }));
        $("device-heading").textContent = "Waiting for the next picture…";
      });
    }
    document.addEventListener("visualassist:camera", (event) => {
      state.camera = event.detail.running;
      if (state.scene === "live") show();
    });
    // A question has been answered: follow it, if that is what is being watched.
    document.addEventListener("visualassist:answer", (event) => {
      state.scenes.live = event.detail;
      if (state.scene !== "live") return;
      choose("live");
      play();
    });
    $("live-inside").addEventListener("click", () => {
      choose("live");
      play();
    });
  }

  fetch("demo/device.json", { cache: "no-cache" })
    .then((response) => {
      if (!response.ok) throw new Error(`device.json: ${response.status}`);
      return response.json();
    })
    .then(start)
    .catch((error) => {
      console.error(error);
      $("device-heading").textContent = "The walkthrough could not be loaded.";
      $("device-body").textContent = "The diagram and the table of parts below still show how the device is put together.";
    });
})();

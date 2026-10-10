# Visual Assist — Navigating the Unseen

**Try the live demo: <https://saravanakumart362-create.github.io/Visual-assist/>**
It runs in your browser with nothing to install: a simulated room, your own
camera, and a walkthrough of the hardware.

A deep learning based aid for the blind. A Raspberry Pi 4 watches the room
through a camera, recognises objects with YOLOv5, measures how far away they
are with a LiDAR, recognises familiar faces and banknotes, and talks to the
user through an earphone. The user talks back: *"Where is the door?"*,
*"Guide me to the table."*

This repository is the software for the B.Tech project report of the same
name (Government College of Engineering Kannur, April 2025).

**Full technical documentation:** [docs/TECHNICAL.md](docs/TECHNICAL.md)
covers the architecture, the wire protocols, the guidance algorithms, every
setting, the models, the demo page and the tests. The same document as a
PDF to download:
[Visual-Assist-Technical-Documentation.pdf](docs/Visual-Assist-Technical-Documentation.pdf).

## What it does

| Feature | How |
|---|---|
| Object detection | YOLOv5 on each camera frame; every object is placed left / centre / right |
| Doors | A second YOLOv5 model, trained for this project (see [Models](#models-and-training)) |
| Distance | TF-Luna LiDAR on the same pan-tilt head as the camera, cross-checked against the object's size in the picture |
| Obstacle warning | Speaks when something is too close: "Bottle detected at 0.4 meters." |
| Find an object | The head sweeps the room, then turns to the object so the LiDAR can range it |
| Guidance | Plans the shortest route around obstacles and speaks it one step at a time |
| Adaptation | Warns earlier when you walk faster, says more when you hesitate, less on familiar trips |
| Faces | Announces registered people, most familiar first; offers to register strangers |
| Banknotes | Recognises ₹10, 20, 50, 100, 200, 500 and 2000 notes |
| Languages | Speaks English, Malayalam, Hindi or Tamil |
| Voice in | Vosk (offline) or Google Speech Recognition (online) |
| Voice out | espeak-ng (offline) or Google Text-to-Speech (online, natural voices) |

## Voice commands

Commands are spoken in English, whatever language the assistant answers in.

| Say | Result |
|---|---|
| "Where is the door?" / "Find my cup" / "How far is the sofa?" | Direction and distance of that object |
| "Guide me to the table" / "Take me to the door" | Step-by-step guidance until arrival |
| "What is around me?" / "Scan" | Describes the room from left to right |
| "How far?" | Distance to whatever is straight ahead |
| "Who is this?" | Names the people in view |
| "Register face" / "Remember this person as Shafeek" | Stores a new face |
| "What note is this?" | Reads a banknote held up to the camera |
| "Speak in Malayalam" / "Hindi" / "Tamil" / "English" | Changes the spoken language |
| "Speak less" / "More detail" / "Normal detail" | Shorter or fuller answers |
| "Speak faster" / "Speak slower" | Speech speed (offline voice) |
| "Warn me earlier" / "Warn me later" | Obstacle warning distance, 25 cm at a time |
| "Repeat" | Says the last sentence again |
| "Stop" | Ends guidance |
| "Help" | Lists the commands |
| "Shut down" | Exits the program |

Words like *desk*, *mug* and *couch* are understood as *table*, *cup* and
*sofa*. If other people's conversation triggers commands, set
`listener.wake_word` in `config.yaml` (for example `assistant`); then only
phrases containing that word are obeyed.

Language, detail level, speed and warning distance are remembered between
sessions in `data/profile.json`.

## Try it without any hardware

```bash
python main.py simulate                  # a chair between the user and the door
python main.py simulate --room opening   # a wall with one opening
python main.py simulate --seed 5         # a random cluttered room
python main.py --language ml simulate    # the same, spoken in Malayalam
```

This runs the real assistant against a simulated room, camera, LiDAR and
pan-tilt head, with a simulated user who does what they are told. It prints
the conversation and a map of the path walked.

## Demo page

`index.html` is a web page with three demonstrations. The first two are
driven by this project's own Python code, run in the browser by
[Pyodide](https://pyodide.org); the third plays a recording made by it:

- **A simulated room.** Visitors drag the door and the obstacles around,
  choose a language, press "Guide me to the door" and watch the route being
  found and spoken. Only the room and the sensors are simulated.
- **Their own camera.** Visitors start their camera and ask "what is around
  me", "where is the door", "what note is this" or "guide me to the chair",
  by typing or (in Chrome and Edge) by voice. The detection, door and
  banknote models run in the page, converted to ONNX and run by
  [ONNX Runtime](https://onnxruntime.ai); the picture never leaves the
  visitor's device. This is what `python main.py --keyboard` does on a
  laptop, with nothing to install.
- **Inside the device.** A diagram of every part and the cable that joins
  it to the Raspberry Pi. Visitors step through "where is the door" or
  "guide me to the door" and see what each part sends to the next: the text
  commands to the ESP8266, the pulses to the servos, the LiDAR's 9 bytes and
  what each of them means. The messages are recorded from the real LiDAR and
  servo drivers (`visual_assist/lidar.py`, `servo.py`) talking to a
  simulated ESP8266 and LiDAR; see `trace()` in `visual_assist/simulation.py`.
  A third choice, "Your camera, live", is not a recording: with the camera
  demonstration running, it follows whatever the visitor asks through the
  same diagram, showing the picture their camera took, what the models
  found in it and the sentence built from that. The parts a phone or laptop
  lacks (LiDAR, ESP8266, servos) are shown faded.

The camera demonstration has no LiDAR and no turning head, so distances are
camera estimates ("about 2 meters") and there is no route planning in it;
face recognition is not included. Its first start downloads about 40 MB.

**Publish it with GitHub Pages:** push the project to GitHub, then in the
repository open *Settings → Pages*, choose *Deploy from a branch*, select
branch `main` and folder `/ (root)`, and save. After a minute the page is at
`https://saravanakumart362-create.github.io/Visual-assist/`.

**Preview it on your own PC:**

```bash
python -m http.server 8000      # then open http://localhost:8000/
```

Opening `index.html` by double-clicking does not work, because browsers do
not let a page read other files straight from the disk.

The two preset rooms play from `demo/recordings.json`, so they start at once
and work even where Pyodide cannot load; the device walkthrough plays from
`demo/device.json`. After changing the guidance code, or the page's style
sheet or scripts, run `python demo/make_recordings.py`; the tests fail until
you do. It refreshes both files and renews the `?v=` codes on the links in
`index.html`, which make browsers fetch the changed files instead of
showing the new page with old ones they kept. Keep the empty `.nojekyll` file: without it GitHub
Pages hides `visual_assist/__init__.py` and the page cannot load the code.

The models the camera demonstration uses are in `demo/models/`. After
retraining a model, or changing a confidence threshold in `config.yaml`,
convert them again:

```bash
pip install onnx onnxslim
python demo/export_models.py
```

## How guidance works

**Finding the target.** The head looks at five angles; whatever the camera
recognises is noted with its direction. The head then turns to the target so
the LiDAR points at it.

**Checking the distance.** The LiDAR measures whatever its beam hits, which
may be a chair in front of the door and not the door. So each reading is
compared with the distance the object's size in the picture implies (a door
is about 2 m tall). If one is more than one and a half times the other, the
beam is on something else: the camera's estimate is used, and spoken as
"about 5 meters".

**Planning the route.** The head sweeps the LiDAR across 144° in 6° steps.
Each hit becomes an obstacle on a small map of the floor (10 cm cells), as
does each object the camera saw. The A* algorithm finds the shortest route
that keeps 35 cm away from all of them.

**Speaking it.** One action at a time: first "Turn slightly left", and only
when the user faces a clear stretch, "Move forward 4 steps, then turn
right". After each action the head looks again and the route is planned
afresh, so imprecise turns and steps are corrected as they happen. Because
people's turns and steps are imprecise, the assistant:

- never asks for a walk longer than what the LiDAR just measured as clear,
  less a 15% margin, and never more than 5 steps at once;
- remembers obstacles while the user turns on the spot (the sweep only sees
  what is in front);
- does not send the user back and forth between two routes of similar length;
- says "I cannot find a clear way to the door" and stops when the way is
  blocked, when the user has turned full circle without finding one, or when
  ten instructions have brought the target no nearer.

Without the LiDAR or the pan-tilt head (on a laptop, say) there is no route
planning; guidance then steers straight at the target by camera alone.

## Feedback and adaptation

The device has no motion sensor, so the user's movement is read from what it
has: walking speed from how fast the LiDAR distance ahead shrinks, turning
from how far the camera picture slides between frames, and standing still
from neither changing.

| The assistant notices | and responds by |
|---|---|
| Walking speed | Warning further ahead: 1 m when still, plus 1.5 s of walking (2.5 m at 1 m/s, 3 m at most) |
| An obstacle less than 1.5 s away | "Stop, chair ahead, 1.1 meters", cutting in on anything being said |
| Standing in front of the same obstacle | Not repeating the warning until something changes |
| A pause of 6 s after walking | During guidance: repeating the instruction |
| Two pauses within two minutes | Describing what is ahead at each pause, and fuller guidance |
| Three or more people in view | Saying the area is crowded, then keeping to short obstacle warnings; strangers are not announced |
| A target guided to three times before | Brief guidance: "Table, slightly to your right, 2.5 meters." "3 steps." |
| A person met often | Announcing them first, and by name only |
| A person not met for a day or more | "Ashna is in front of you. You last met 3 days ago." |
| The user's usual walking pace | Timing the wait after "move forward" to match |
| Spoken preferences | Language, detail, speed and warning distance, kept between sessions |

Speed can only be measured while there is something within the LiDAR's 8 m
to walk towards.

## Languages

Every sentence is stored per language in `visual_assist/locales/`
(`en.json`, `ml.json`, `hi.json`, `ta.json`), with the grammar each language
needs (Hindi verb gender and noun endings, Tamil's separate verb for
people). Set `speech.language` in `config.yaml`, start with
`--language ml`, or say "speak in Malayalam".

English uses the offline voice. For the other three the natural Google
voice is used when the internet is reachable, and the offline espeak-ng
voice when it is not, so a dropped connection never means silence.

The Malayalam, Hindi and Tamil sentences were written by a non-native
speaker and should be read through by a native speaker before real use;
each is one line in the JSON file. To add a language, copy `en.json` and
translate the right-hand sides.

## Hardware

- Raspberry Pi 4 with Raspberry Pi OS (64-bit)
- Raspberry Pi Camera Module (CSI ribbon)
- TF-Luna LiDAR
- Pan-tilt bracket with two SG90 servos, driven by an ESP8266 (NodeMCU)
- USB microphone, earphone on the 3.5 mm jack

### Wiring

**TF-Luna → Raspberry Pi (UART)**

| TF-Luna pin | Raspberry Pi |
|---|---|
| 1 — +5 V | Pin 2 (5 V) |
| 2 — RXD | Pin 8 (GPIO14, TXD) |
| 3 — TXD | Pin 10 (GPIO15, RXD) |
| 4 — GND | Pin 6 (GND) |
| 5, 6 | Not connected |

**Servos → ESP8266** (the ESP8266 plugs into a Pi USB port)

| Servo wire | Connect to |
|---|---|
| Pan signal (orange) | D5 |
| Tilt signal (orange) | D6 |
| Power (red) | 5 V supply — not the ESP8266's 3.3 V pin |
| Ground (brown) | GND, shared with the ESP8266 |

Mount the LiDAR so it points the same way as the camera: the software assumes
the LiDAR measures along the centre of the picture.

If the servos are wired straight to the Pi instead, set `servo.backend: gpio`
and the `pan_pin` / `tilt_pin` BCM numbers in `config.yaml`.

## Setup on the Raspberry Pi

```bash
# 1. Enable the camera and the serial port (Interface Options ->
#    Serial Port: login shell NO, hardware port YES), then reboot.
sudo raspi-config

# 2. System packages
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv espeak-ng mpg123 \
                    portaudio19-dev cmake

# 3. Python packages (the venv must see the apt-installed picamera2)
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Offline speech recognition model
mkdir -p models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip && cd ..
```

Copy the whole project folder to the Pi, including `weights/doors.pt` and
`weights/currency.pt`. Flash `firmware/esp8266_pan_tilt/esp8266_pan_tilt.ino`
to the ESP8266 with the Arduino IDE (board: *NodeMCU 1.0*), then plug it into
the Pi.

## Running

```bash
python main.py check             # which devices and libraries were found
python main.py                   # start the assistant
python main.py --keyboard        # type commands instead of speaking
python main.py --show            # also show the camera view with boxes
python main.py --language hi     # speak Hindi
python main.py register Ashna    # register the face in front of the camera
```

Always run `check` first. Every device is optional: a missing LiDAR means
distances are camera estimates, missing servos mean no room sweep or route
planning, a missing microphone falls back to typed commands. This is also
how to run it on a laptop with a webcam (`pip install opencv-python
ultralytics pyyaml pyserial gTTS`).

If "turn right" guidance sends the head left, set `servo.pan_invert: true`.

## Models and training

| File | What | Trained on |
|---|---|---|
| `weights/yolov5nu.pt` | 80 everyday objects (chair, sofa, table, bottle, cup, person, …) | COCO, by Ultralytics; downloaded on first run |
| `weights/doors.pt` | Doors | DoorDetect + Open Images, by `training/train.py doors` |
| `weights/currency.pt` | Indian banknotes | Mendeley banknote dataset, by `training/train.py currency` |

### How good the door model is

YOLOv5-nano, trained for 30 epochs on 1,715 photos (1,451 labelled doors)
and measured on 229 photos it never saw. By the usual strict score its door
AP50 is 0.45. Judged the way the assistant uses it (does it point at a real
door?), at the confidence set in `config.yaml`:

| | Indoor photos (136) | All photos (229) |
|---|---|---|
| When it reports a door, it is right | 69% (38 of 55) | 71% (102 of 143) |
| Doors found | 64% (39 of 61) | 54% (98 of 181) |
| Large doors found (at least 8% of the picture) | 85% (17 of 20) | 67% (53 of 79) |
| Photos with a false door | 15 of 136 | 32 of 229 |

That is enough for a prototype and well ahead of the ready-made
alternative (Ultralytics' Open Images models found 7–20% of the same indoor
doors), but it is not reliable: small, distant or half-hidden doors are
often missed, and about one photo in nine shows a false door. The assistant
allows for this by scanning from five angles and trusting the most
confident sighting. This model was trained on a laptop CPU with most of the
network frozen; the full training below should do noticeably better.

### How good the banknote model is

A YOLOv8-nano image classifier with 11 classes (old and new series
separately, plus "no note"), trained on 2,575 photos and measured on 625 it
never saw:

| Held-out photos | Result |
|---|---|
| 400 banknotes | 390 named correctly and stated plainly |
| | 9 named correctly, offered as a guess ("…but I am not sure. Please show it again.") |
| | 1 named wrongly, offered as a guess |
| | 0 named wrongly and stated plainly; 0 missed |
| 225 photos with no banknote (faces, hands, rooms, streets) | 225 correctly reported as no note |

A value is stated plainly only above 0.8 confidence, because a confident
mistake about money is the costly kind; between 0.5 and 0.8 it is offered
as a guess, and below 0.5 it is treated as no note.

The banknote photos come from the same collection as the training photos
(same notes, same camera), so expect lower accuracy on a live camera. It
has not been tried with real notes. The "no note" class matters as much as
the notes: a camera held by a person mostly sees that person, so the class
includes several hundred photos of faces and hands.

### Training again

```bash
pip install ultralytics pillow
python training/prepare_doors.py       # downloads ~700 MB, builds the dataset
python training/train.py doors         # about 1.5 hours on a laptop CPU
python training/prepare_currency.py    # downloads ~750 MB
python training/train.py currency      # about ten minutes
```

The defaults are sized for a CPU. On Google Colab with a GPU, as in the
report, train the door model fully for a better result:

```bash
python training/train.py doors --imgsz 640 --epochs 100 --freeze 0
```

To train on your own labelled photos (YOLO format; see
`training/dataset.yaml`), run `python training/train.py custom --data
my.yaml --name indoor` and add `weights/indoor.pt` to `detector.extra` in
`config.yaml`.

### Data sources

- **DoorDetect** — M. Arduengo, C. Torras, L. Sentis, "Robust and adaptive
  door operation with a mobile robot", *Intelligent Service Robotics*, 2021.
  <https://github.com/MiguelARD/DoorDetect-Dataset>
- **Open Images V7** — Google; images CC BY 2.0, annotations CC BY 4.0.
  <https://storage.googleapis.com/openimages/web/index.html>
- **Dataset of Indian (Rupees) and Thai (Baht) Banknotes with Annotations**
  — V. Meshram, K. Patil, P. Chumchu, Mendeley Data, 2021, CC BY 4.0.
  <https://data.mendeley.com/datasets/2kfz5yc7pt/1>

## Tests

```bash
python -m pytest
```

The tests need no hardware. Besides the building blocks (commands, LiDAR
frames, planner, phrases in all four languages, adaptation) they run the
whole assistant in simulated rooms: 240 random cluttered rooms are walked,
half of them by a simulated user whose turns are off by up to a third and
whose steps by up to 15%. The tests require that nobody is ever walked
into anything, and that at least 9 trips in 10 reach the door.

## Project layout

```
main.py                      command line entry point
config.yaml                  all settings
docs/TECHNICAL.md            full technical documentation
index.html, demo/            the demo page (see "Demo page")
visual_assist/
  assistant.py               main loop, voice commands, guidance
  planner.py                 obstacle map and A* route search
  navigator.py               bearings, distance checks, next instruction
  adaptation.py              motion tracking, user profile, adaptive rules
  phrases.py, locales/       every spoken sentence, in four languages
  camera.py                  Pi Camera / webcam
  detector.py                YOLOv5 detection, one model or several
  currency.py                banknote recognition
  lidar.py                   TF-Luna frame decoding
  servo.py                   pan-tilt head (ESP8266 or GPIO)
  commands.py                speech text -> intent
  speech.py                  text-to-speech
  listener.py                speech-to-text
  faces.py                   face recognition and registration
  simulation.py              simulated room and hardware (tests, simulate)
  live.py                    the assistant fed by a web page's camera
firmware/esp8266_pan_tilt/   Arduino sketch for the servo controller
training/                    dataset preparation and training scripts
weights/                     the trained models
tests/                       python -m pytest
```

## What has and has not been verified

The report describes the system but does not include its source code, so
this is a fresh implementation of that description.

**Verified on a PC:** the test suite; guidance in simulation (997 of 1,000
random rooms reached, the other three ending with a spoken stop, nothing
ever touched); the door and banknote models on held-out photos (figures
above); speech synthesis in all four languages; a short live trial on a
webcam with typed questions; and the demo page in a headless browser, its
camera part with saved photos standing in for the camera. The camera part
has not been tried on a real camera or on a phone. The LiDAR and servo
drivers have been run only against the simulated LiDAR and ESP8266 that the
device walkthrough uses.

**Not verified:** anything on the Raspberry Pi itself — the Pi Camera, the
TF-Luna, the servos and ESP8266, the microphone, and face recognition (its
library is not installed on the development PC). Expect to adjust
`config.yaml` when the hardware is first connected: serial ports,
`pan_invert`, and the camera's field of view.

**Known limits**

- Route planning has been exercised only against simulated rooms. Real
  LiDAR readings are noisier, and a single beam at one height does not see
  obstacles above or below it; the white cane stays essential.
- The camera's distance estimate assumes ordinary sizes (a 2 m door, a
  0.9 m chair). Tables and sofas vary too much, so without the LiDAR their
  distance is not given.
- Walking speed is measured only while walking towards something within
  the LiDAR's 8 m.
- Voice commands are English only; the spoken replies are multilingual.
- Banknote recognition expects the note held close to the camera, filling
  most of the picture, as in the training photos.

**Choices made where the report is silent or inconsistent**

- *LiDAR model.* The text names both "TF-Luna" and "TF-LC02"; the code
  screenshot (Fig. 4.7) reads TF-Luna UART frames, so that is what is
  implemented. The circuit diagram labels the LiDAR lines SDA/SCL (I²C), but
  the same screenshot uses serial, so UART wiring is given above.
- *Servo control.* The text says the ESP8266 drives the servos; the circuit
  diagram wires them to the Pi. Both are supported.
- *Two detection models.* The report's custom model covered doors, chairs
  and desks. Here the pretrained model handles everyday objects and a
  separately trained one handles doors, which keeps the pretrained model's
  accuracy on the former.

Face data stays on the device: only numeric encodings are stored, in
`data/faces/faces.json`.

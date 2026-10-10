"""Draw the technical drawings of docs/TECHNICAL.md.

    python docs/make_figures.py

Writes docs/img/*.svg. The numbers in them are not typed in: each drawing
is computed from the project's own code and settings (the planner, the
LiDAR frame encoder, the simulator, the default configuration), so run
this again after changing any of those.
"""

import html
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from visual_assist import adaptation, config, lidar, navigator, planner  # noqa: E402
from visual_assist.simulation import ROOMS, World, record, trace  # noqa: E402

OUT = HERE / "img"
CFG = config.from_dict({})
ENGLISH = json.loads((HERE.parent / "visual_assist" / "locales" / "en.json")
                     .read_text(encoding="utf-8"))["phrases"]

INK, MUTED, LINE, WASH = "#13202c", "#53626f", "#c5ced6", "#eef1f4"
ACCENT, PATH, LIDAR, SOLID = "#f5b800", "#0c7c74", "#cf2e45", "#44556a"
FONT = "Segoe UI, Helvetica, Arial, sans-serif"
MONO = "Consolas, Menlo, monospace"
ARROWS = {"ink": INK, "path": PATH, "lidar": LIDAR, "muted": MUTED, "blue": "#1f5fae"}


# ------------------------------------------------------------ drawing helpers

def n(value: float) -> str:
    """A coordinate, written short."""
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _style(fill=None, stroke=None, width=None, dash=None, opacity=None, marker=None,
           start=None) -> str:
    parts = [f'fill="{fill or "none"}"']
    if stroke:
        parts.append(f'stroke="{stroke}" stroke-width="{width or 1.5}"')
    if dash:
        parts.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        parts.append(f'opacity="{opacity}"')
    if marker:
        parts.append(f'marker-end="url(#arrow-{marker})"')
    if start:
        parts.append(f'marker-start="url(#arrow-{start})"')
    return " ".join(parts)


def text(x, y, content, anchor="start", size=13, fill=INK, weight=None, family=None,
         halo=False) -> str:
    """One line of text. `halo` gives it a white edge, to stay readable over lines."""
    attrs = [f'x="{n(x)}" y="{n(y)}" font-size="{size}" fill="{fill}"']
    if anchor != "start":
        attrs.append(f'text-anchor="{anchor}"')
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if family:
        attrs.append(f'font-family="{family}"')
    if halo:
        attrs.append('stroke="#ffffff" stroke-width="4" paint-order="stroke" '
                     'stroke-linejoin="round"')
    return f"<text {' '.join(attrs)}>{html.escape(str(content))}</text>"


def lines(x, y, rows, leading=17, **style) -> str:
    """Several lines of text, one under the other."""
    return "".join(text(x, y + i * leading, row, **style) for i, row in enumerate(rows))


def line(x1, y1, x2, y2, stroke=INK, **style) -> str:
    return (f'<line x1="{n(x1)}" y1="{n(y1)}" x2="{n(x2)}" y2="{n(y2)}" '
            f'{_style(stroke=stroke, **style)}/>')


def circle(cx, cy, r, **style) -> str:
    return f'<circle cx="{n(cx)}" cy="{n(cy)}" r="{n(r)}" {_style(**style)}/>'


def rect(x, y, w, h, rx=4, **style) -> str:
    return (f'<rect x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}" rx="{rx}" '
            f'{_style(**style)}/>')


def path(d, **style) -> str:
    return f'<path d="{d}" {_style(**style)}/>'


def polyline(points, **style) -> str:
    joined = " ".join(f"{n(x)},{n(y)}" for x, y in points)
    return f'<polyline points="{joined}" {_style(**style)}/>'


def badge(x, y, label, fill=INK) -> str:
    """A numbered marker."""
    return circle(x, y, 10, fill=fill) + text(x, y + 4.5, label, "middle", 12, "#ffffff", "700")


def figure(width, height, label, parts) -> str:
    """A whole drawing on a white ground, so it reads on a light or a dark page."""
    markers = "".join(
        f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" '
        f'fill="{colour}"/></marker>' for name, colour in ARROWS.items())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" font-family="{FONT}" '
            f'aria-label="{html.escape(label)}">\n<title>{html.escape(label)}</title>\n'
            f'<defs>{markers}</defs>\n<rect width="{width}" height="{height}" fill="#ffffff"/>\n'
            + "\n".join(parts) + "\n</svg>\n")


class Top:
    """A view from above. The wearer is at (ox, oy) facing up the page;
    `scale` is drawing units to a metre."""

    def __init__(self, ox: float, oy: float, scale: float):
        self.ox, self.oy, self.scale = ox, oy, scale

    def xy(self, x: float, y: float):
        return self.ox + x * self.scale, self.oy - y * self.scale

    def polar(self, bearing: float, distance: float):
        angle = math.radians(bearing)
        return self.xy(distance * math.sin(angle), distance * math.cos(angle))

    def arc(self, first: float, last: float, distance: float, **style) -> str:
        (x1, y1), (x2, y2) = self.polar(first, distance), self.polar(last, distance)
        r = distance * self.scale
        big = 1 if last - first > 180 else 0
        return path(f"M{n(x1)} {n(y1)} A{n(r)} {n(r)} 0 {big} 1 {n(x2)} {n(y2)}", **style)

    def wedge(self, first: float, last: float, distance: float, **style) -> str:
        (x1, y1), (x2, y2) = self.polar(first, distance), self.polar(last, distance)
        r = distance * self.scale
        big = 1 if last - first > 180 else 0
        return path(f"M{n(self.ox)} {n(self.oy)} L{n(x1)} {n(y1)} "
                    f"A{n(r)} {n(r)} 0 {big} 1 {n(x2)} {n(y2)} Z", **style)

    def disc(self, x: float, y: float, radius: float, **style) -> str:
        cx, cy = self.xy(x, y)
        return circle(cx, cy, radius * self.scale, **style)


def side(bearing: float) -> str:
    return "ahead" if bearing == 0 else f"{abs(bearing):g}° {'left' if bearing < 0 else 'right'}"


# ------------------------------------------------------------------ drawings

def wiring() -> str:
    """Every wire between the parts."""
    red, black, orange, blue, cable = "#c62828", "#222222", "#e07b00", "#1f5fae", "#8a97a3"
    p = []

    def terminal(x, y, label, anchor="start", gap=12):
        offset = gap if anchor == "start" else -gap
        return circle(x, y, 4, fill=INK) + text(x + offset, y + 4, label, anchor, 12)

    def ground(x, y):
        return "".join(line(x - half, y + drop, x + half, y + drop, stroke=black, width=2)
                       for half, drop in ((9, 0), (6, 4), (3, 8)))

    # The Raspberry Pi, in the middle.
    p.append(rect(310, 30, 180, 410, fill=WASH, stroke=INK))
    p.append(text(400, 58, "Raspberry Pi 4", "middle", 15, weight="700"))
    for y, label in ((100, "Pin 2: 5 V"), (135, "Pin 8: TXD (GPIO14)"),
                     (170, "Pin 10: RXD (GPIO15)"), (205, "Pin 6: GND"),
                     (290, "CSI connector"), (340, "USB port"), (390, "3.5 mm audio jack")):
        p.append(terminal(310, y, label))
    p.append(terminal(490, 100, "USB port", "end"))

    # The LiDAR, on four wires.
    p.append(rect(30, 45, 170, 205, fill="#ffffff", stroke=INK))
    p.append(text(115, 72, "TF-Luna LiDAR", "middle", 14, weight="700"))
    for y, label, colour, note, dash in ((100, "1: +5 V", red, "5 V", None),
                                         (135, "2: RXD", blue, "not used", "5 4"),
                                         (170, "3: TXD", blue, "readings", None),
                                         (205, "4: GND", black, "ground", None)):
        arrow = "blue" if note == "readings" else None
        p.append(line(200, y, 306 if arrow else 310, y, stroke=colour, width=2.5, dash=dash,
                      marker=arrow))
        p.append(terminal(200, y, label, "end"))
        p.append(text(255, y - 7, note, "middle", 11, MUTED))
    p.append(text(188, 236, "5, 6: not connected", "end", 11, MUTED))

    # Camera, microphone, earphone: one cable each.
    for y, label, note in ((290, "Pi Camera", "ribbon cable"), (340, "USB microphone", "USB"),
                           (390, "Earphone", "audio cable")):
        p.append(line(200, y, 310, y, stroke=cable, width=4))
        p.append(rect(30, y - 18, 170, 36, fill="#ffffff", stroke=INK))
        p.append(text(115, y + 5, label, "middle", 13, weight="600"))
        p.append(text(255, y - 8, note, "middle", 11, MUTED))

    # The ESP8266, on a USB cable.
    p.append(line(490, 100, 600, 100, stroke=cable, width=4))
    p.append(text(545, 91, "USB cable", "middle", 11, MUTED))
    p.append(text(545, 118, "power and serial", "middle", 11, MUTED))
    p.append(rect(600, 45, 170, 175, fill="#ffffff", stroke=INK))
    p.append(text(685, 72, "ESP8266 (NodeMCU)", "middle", 14, weight="700"))
    p.append(terminal(600, 100, "USB"))
    for x, label in ((640, "D5"), (690, "D6"), (740, "GND")):
        p.append(circle(x, 220, 4, fill=INK))
        p.append(text(x, 207, label, "middle", 12))

    # The two servos: signal from the ESP8266, power from a supply of their own.
    p.append(path("M640 220 V262 H615 V300", stroke=orange, width=2.5))
    p.append(path("M690 220 V275 H735 V300", stroke=orange, width=2.5))
    p.append(text(607, 291, "signal", "end", 11, MUTED))
    p.append(text(743, 291, "signal", "start", 11, MUTED))
    p.append(line(740, 220, 740, 234, stroke=black, width=2) + ground(740, 234))
    for x, label, plus, minus in ((570, "Pan servo", 595, 635), (690, "Tilt servo", 715, 755)):
        p.append(rect(x, 300, 90, 60, fill="#ffffff", stroke=INK))
        p.append(text(x + 45, 335, label, "middle", 13, weight="600"))
        p.append(circle(x + 45, 300, 4, fill=INK))
        p.append(circle(plus, 360, 4, fill=INK) + circle(minus, 360, 4, fill=INK))
        p.append(line(minus, 360, minus, 372, stroke=black, width=2) + ground(minus, 372))
    p.append(path("M595 360 V394 H715 V360 M655 394 V414", stroke=red, width=2.5))
    p.append(text(675, 388, "+5 V", "middle", 11, MUTED, halo=True))
    p.append(rect(590, 414, 130, 34, fill="#ffffff", stroke=INK))
    p.append(text(655, 436, "5 V supply", "middle", 13, weight="600"))
    p.append(circle(655, 414, 4, fill=INK))
    p.append(path("M720 431 H742 V438", stroke=black, width=2) + ground(742, 438))
    p.append(lines(468, 474, ["Servo power comes from its own 5 V supply, not from the",
                              "ESP8266's 3.3 V pin. All the ground points are joined."],
                   leading=15, size=11.5, fill=MUTED))
    return figure(800, 500, "Wiring: the TF-Luna on four wires to the Raspberry Pi's pins 2, 8, "
                  "10 and 6; the camera, microphone, earphone and ESP8266 on one cable each; "
                  "the servo signals on the ESP8266's D5 and D6, with power from a separate "
                  "5 V supply.", p)


def lidar_frame() -> str:
    """The 9 bytes of one reading, as the encoder writes them."""
    metres = 2.26
    frame = lidar.encode_frame(metres)
    assert lidar.find_frame(frame) == frame and lidar.parse_frame(frame) == metres
    cells = frame.hex(" ").upper().split()
    distance, strength = frame[2] + frame[3] * 256, frame[4] + frame[5] * 256
    groups = [
        (0, 2, WASH, ["Header", "always 59 59"]),
        (2, 2, "#fde9b0", ["Distance, low byte first", f"0x{distance:04X} = {distance} cm"]),
        (4, 2, "#cfe6e3", ["Signal strength", f"0x{strength:04X} = {strength}"]),
        (6, 2, WASH, ["Temperature", "not used"]),
        (8, 1, "#f6cfd5", ["Checksum", "low byte of the sum", "of bytes 1 to 8"]),
    ]
    x0, y0, w, h = 44, 46, 68, 48
    p = []
    for first, count, fill, label in groups:
        left, right = x0 + first * w, x0 + (first + count) * w
        p.append(rect(left, y0, right - left, h, rx=0, fill=fill, stroke=INK))
        p.append(path(f"M{left + 4} {y0 + h + 8} V{y0 + h + 15} H{right - 4} V{y0 + h + 8}",
                      stroke=MUTED))
        middle = (left + right) / 2
        p.append(text(middle, y0 + h + 36, label[0], "middle", 13, weight="700"))
        p.append(lines(middle, y0 + h + 54, label[1:], leading=16, anchor="middle", size=12,
                       fill=MUTED))
    for index, cell in enumerate(cells):
        middle = x0 + (index + 0.5) * w
        if index:
            p.append(line(x0 + index * w, y0, x0 + index * w, y0 + h, stroke=INK, width=0.75))
        p.append(text(middle, y0 - 10, index + 1, "middle", 11, MUTED))
        p.append(text(middle, y0 + 31, cell, "middle", 20, family=MONO, weight="700"))
    p.append(text(x0 - 8, y0 - 10, "byte", "end", 11, MUTED))
    return figure(700, 196, f"The nine bytes of a TF-Luna frame for a reading of {metres} m: "
                  f"{' '.join(cells)}.", p)


def camera_geometry() -> str:
    """How a pixel gives a direction, and an apparent size a distance."""
    cam = CFG.camera
    p = []

    # ---- left: bearing from a pixel, seen from above
    cx_pixel = 480
    bearing = navigator.pixel_bearing(cx_pixel, cam.width, cam.hfov_deg)
    ox, oy, top = 195, 300, 96
    half = (oy - top) * math.tan(math.radians(cam.hfov_deg / 2))
    spot = ox + (cx_pixel - cam.width / 2) / (cam.width / 2) * half
    p.append(text(20, 30, "Direction from a pixel", size=15, weight="700"))
    p.append(text(20, 49, "seen from above", size=12, fill=MUTED))
    p.append(path(f"M{ox} {oy} L{n(ox - half)} {top} L{n(ox + half)} {top} Z", fill=WASH))
    p.append(line(ox, oy, ox, top, stroke=MUTED, dash="5 4"))
    p.append(line(ox - half, top, ox + half, top, stroke=INK, width=3))
    for x, label in ((ox - half, "0"), (ox, str(cam.width // 2)), (ox + half, str(cam.width))):
        p.append(line(x, top - 5, x, top + 5, stroke=INK))
        p.append(text(x, top - 11, label, "middle", 12, MUTED))
    p.append(text(ox, top - 32, f"the picture: {cam.width} pixels wide", "middle", 12))
    p.append(line(ox, oy, spot, top, stroke=PATH, width=2.5))
    p.append(circle(spot, top, 5, fill=PATH))
    p.append(text(spot, top + 22, f"cx = {cx_pixel}", "middle", 12, PATH, "700", halo=True))
    view = Top(ox, oy, 1)
    p.append(view.arc(0, bearing, 78, stroke=PATH, width=1.5))
    p.append(text(ox + 38, oy - 72, f"{bearing:.1f}°", size=13, fill=PATH, weight="700"))
    p.append(view.arc(-cam.hfov_deg / 2, cam.hfov_deg / 2, 36, stroke=MUTED))
    p.append(circle(ox, oy, 6, fill=INK))
    p.append(text(ox, oy + 24, f"camera, field of view {cam.hfov_deg}°", "middle", 12, MUTED))

    # ---- right: distance from apparent size, seen from the side
    label, real = "door", navigator.TYPICAL_SIZE_M["door"][0]
    tall = 240
    box = (270, (cam.height - tall) / 2, 370, (cam.height + tall) / 2)
    distance = navigator.visual_distance(label, box, cam.width, cam.height, cam.hfov_deg,
                                         cam.vfov_deg)
    focal = (cam.height / 2) / math.tan(math.radians(cam.vfov_deg / 2))
    eye_x, eye_y, plane, far = 440, 178, 512, 742
    unit = (plane - eye_x) / focal  # drawing units to a pixel of the picture
    shown, whole = tall / 2 * unit, cam.height / 2 * unit
    reach = shown * (far - eye_x) / (plane - eye_x)
    p.append(text(420, 30, "Distance from apparent size", size=15, weight="700"))
    p.append(text(420, 49, "seen from the side", size=12, fill=MUTED))
    p.append(path(f"M{eye_x} {eye_y} L{far} {n(eye_y - reach)} L{far} {n(eye_y + reach)} Z",
                  fill=WASH))
    p.append(line(eye_x, eye_y, far, eye_y, stroke=MUTED, dash="5 4"))
    p.append(line(eye_x, eye_y, far, eye_y - reach, stroke=PATH))
    p.append(line(eye_x, eye_y, far, eye_y + reach, stroke=PATH))
    p.append(line(plane, eye_y - whole, plane, eye_y + whole, stroke=INK, width=3))
    p.append(line(plane, eye_y - shown, plane, eye_y + shown, stroke=PATH, width=6))
    p.append(lines(plane, eye_y - whole - 26, ["the picture:", f"{cam.height} px tall"],
                   leading=15, anchor="middle", size=12))
    p.append(text(plane, eye_y + whole + 18, f"{label}: {tall} px", "middle", 12, PATH, "700"))
    p.append(rect(far - 4, eye_y - reach, 8, 2 * reach, rx=0, fill=ACCENT, stroke=INK,
                  width=1))
    p.append(text(far, eye_y - reach - 10, f"{label}, {real:.1f} m tall", "middle", 12))
    p.append(circle(eye_x, eye_y, 6, fill=INK))
    p.append(text(eye_x, eye_y - 14, "camera", "middle", 12, MUTED))
    for y, right, note in ((274, plane, f"f = {focal:.0f} px"),
                           (304, far, f"d = {real:.1f} m × {focal:.0f} / {tall} = {distance:.1f} m")):
        p.append(line(eye_x, y, right, y, stroke=INK, marker="ink", start="ink"))
        p.append(text((eye_x + right) / 2, y + 17, note, "middle", 12.5,
                      weight="700" if right == far else None))
    return figure(800, 340, f"Left: a pixel at column {cx_pixel} of {cam.width} lies "
                  f"{bearing:.1f} degrees right of the camera's axis. Right: a {real:.1f} m "
                  f"{label} that is {tall} pixels tall in a {cam.height}-pixel picture is "
                  f"{distance:.1f} m away.", p)


def beam_blocked() -> str:
    """Why the LiDAR's reading is checked against the camera's estimate."""
    aim = next(step for step in trace("guide")["steps"] if step["kind"] == "aim")
    beam, looks, target = aim["metres"], aim["looks"], aim["target"]
    chair = ROOMS["chair"][0]["chair"]
    width = navigator.TYPICAL_SIZE_M[target][1]
    scale, ox, oy = 106, 70, 100

    def at(metres: float) -> float:
        return ox + metres * scale

    p = []
    door_x, half = at(looks), width / 2 * scale
    p.append(path(f"M{ox} {oy} L{n(door_x)} {n(oy - half)} L{n(door_x)} {n(oy + half)} Z",
                  fill=WASH))
    p.append(line(ox, oy, door_x, oy - half, stroke=PATH, dash="5 4"))
    p.append(line(ox, oy, door_x, oy + half, stroke=PATH, dash="5 4"))
    p.append(circle(at(chair[1]), oy, chair[2] * scale, fill=SOLID))
    p.append(text(at(chair[1]), oy - chair[2] * scale - 10, "chair", "middle", 13, weight="700"))
    p.append(rect(door_x, oy - half, 9, 2 * half, rx=0, fill=ACCENT, stroke=INK, width=1))
    p.append(text(door_x + 4, oy - half - 10, target, "middle", 13, weight="700"))
    p.append(line(ox, oy, at(beam), oy, stroke=LIDAR, width=3))
    p.append(circle(at(beam), oy, 5, fill=LIDAR))
    p.append(circle(ox, oy, 7, fill=PATH))
    p.append(text(ox, oy + 26, "wearer", "middle", 12, MUTED))
    for y, metres, colour, note in (
            (180, beam, LIDAR, f"LiDAR: the beam stops at the chair, {beam:.2f} m"),
            (214, looks, PATH, f"Camera: by its size in the picture, the {target} is "
                               f"about {looks:g} m away")):
        p.append(line(ox, y, at(metres), y, stroke=colour, marker=None, width=1.5))
        p.append(line(ox, y - 5, ox, y + 5, stroke=colour))
        p.append(line(at(metres), y - 5, at(metres), y + 5, stroke=colour))
        p.append(text(ox + 6, y - 7, note, size=12.5, fill=colour, weight="700"))
    ratio = looks / beam
    p.append(text(ox, 250, f"{looks:g} m is {ratio:.1f} times {beam:.2f} m, more than the "
                  f"{navigator.AGREEMENT:g} allowed: the beam is on something else, so the "
                  "camera's estimate is used.", size=12.5))
    return figure(750, 268, f"Seen from above: the LiDAR beam aimed at the {target} stops at "
                  f"a chair {beam:.2f} m away, while the {target}'s size in the picture puts "
                  f"it about {looks:g} m away.", p)


def scan_and_sweep() -> str:
    """What a scan and a sweep cover."""
    servo, plan, cam = CFG.servo, CFG.planner, CFG.camera
    p = []

    left = Top(205, 262, 24)
    reach = plan.max_range_m
    p.append(text(205, 28, f"Scan: {len(servo.scan_angles)} pictures, each "
                  f"{cam.hfov_deg}° wide", "middle", 15, weight="700"))
    for angle in servo.scan_angles:
        p.append(left.wedge(angle - cam.hfov_deg / 2, angle + cam.hfov_deg / 2, reach,
                            fill=PATH, opacity=0.13))
    for angle in servo.scan_angles:
        p.append(left.wedge(angle - cam.hfov_deg / 2, angle + cam.hfov_deg / 2, reach,
                            stroke=PATH, width=0.75))
        x, y = left.polar(angle, reach)
        p.append(line(left.ox, left.oy, x, y, stroke=PATH, dash="5 4"))
        x, y = left.polar(angle, reach + 0.55)
        anchor = "middle" if angle == 0 else "end" if angle < 0 else "start"
        pan = servo.pan_center - angle
        p.append(text(x, y - 8, side(angle), anchor, 12, weight="700"))
        p.append(text(x, y + 7, f"PAN {pan}", anchor, 11.5, MUTED, family=MONO))
    p.append(circle(left.ox, left.oy, 6, fill=INK))
    cover = max(servo.scan_angles) + cam.hfov_deg / 2
    p.append(text(205, 290, f"Together they cover {cover:.0f}° to each side.", "middle", 12,
                  MUTED))

    right = Top(600, 262, 24)
    sensor = lidar.MAX_RANGE_M
    bearings = []
    bearing = -plan.sweep_half_angle_deg
    while bearing <= plan.sweep_half_angle_deg + 1e-6:
        bearings.append(bearing)
        bearing += plan.sweep_step_deg
    p.append(text(600, 28, f"Sweep: {len(bearings)} LiDAR readings, "
                  f"{plan.sweep_step_deg:g}° apart", "middle", 15, weight="700"))
    edge = plan.sweep_half_angle_deg + 6
    for bearing in bearings:
        x1, y1 = right.polar(bearing, reach)
        x2, y2 = right.polar(bearing, sensor)
        p.append(line(x1, y1, x2, y2, stroke=LINE, width=1))
        p.append(line(right.ox, right.oy, x1, y1, stroke=LIDAR, width=1.25))
    p.append(right.arc(-edge, edge, reach, stroke=INK, width=1, dash="4 3"))
    p.append(right.arc(-edge, edge, sensor, stroke=MUTED, width=1, dash="4 3"))
    for metres, note in ((reach, "used by the planner"), (sensor, "the sensor's limit")):
        x, y = right.polar(0, metres)
        p.append(text(x, y - 5, f"{metres:g} m: {note}", "middle", 11.5, halo=True))
    for bearing in (bearings[0], bearings[-1]):
        x, y = right.polar(bearing, reach + 0.45)
        p.append(text(x, y + 16, side(bearing), "end" if bearing < 0 else "start", 12,
                      weight="700", halo=True))
    p.append(circle(right.ox, right.oy, 6, fill=INK))
    p.append(text(600, 290, "No pictures are taken during a sweep.", "middle", 12, MUTED))
    return figure(820, 304, f"Seen from above. Left: the {len(servo.scan_angles)} camera "
                  f"views of a scan, from {side(servo.scan_angles[0])} to "
                  f"{side(servo.scan_angles[-1])}. Right: the {len(bearings)} LiDAR readings "
                  f"of a sweep, from {side(bearings[0])} to {side(bearings[-1])}.", p)


def readings_to_obstacles() -> str:
    """A sweep of the room with a wall and one opening, turned into obstacles."""
    plan = CFG.planner
    things, blocks = ROOMS["opening"]
    world = World(things, blocks)
    rays = []
    bearing = -plan.sweep_half_angle_deg
    while bearing <= plan.sweep_half_angle_deg + 1e-6:
        rays.append((bearing, world.ray(bearing)))
        bearing += plan.sweep_step_deg
    obstacles = planner.obstacles_from_rays(rays, plan.sweep_step_deg, plan.max_range_m)
    hits = {planner.to_xy(b, d) for b, d in rays if d is not None and d <= plan.max_range_m}

    view = Top(232, 398, 62)
    p = ['<clipPath id="room"><rect x="12" y="14" width="440" height="396"/></clipPath>',
         '<g clip-path="url(#room)">']
    for x, y, radius in blocks:
        p.append(view.disc(x, y, radius, fill="#dfe5ea"))
    door = things["door"]
    p.append(view.disc(*door, fill=ACCENT, stroke=INK, width=1))
    for bearing, distance in rays:
        missed = distance is None or distance > plan.max_range_m
        x, y = view.polar(bearing, plan.max_range_m if missed else distance)
        p.append(line(view.ox, view.oy, x, y, stroke=LINE if missed else LIDAR,
                      width=1, opacity=None if missed else 0.55))
    for obstacle in obstacles:
        joined = (obstacle.x, obstacle.y) not in hits
        p.append(view.disc(obstacle.x, obstacle.y, obstacle.radius,
                           fill="#9fb0bf" if joined else SOLID,
                           stroke="#ffffff", width=0.75))
    for x, y in hits:
        px, py = view.xy(x, y)
        p.append(circle(px, py, 2.5, fill=LIDAR))
    p.append(circle(view.ox, view.oy, 6, fill=INK))
    p.append("</g>")
    p.append(rect(12, 14, 440, 396, rx=4, stroke=LINE, width=1))
    x, y = view.xy(door[0], door[1])
    p.append(text(x, y + 4, "door", "middle", 12, weight="700"))

    joins = sum((o.x, o.y) not in hits for o in obstacles)
    notes = [
        ((-0.78, 1.92), ["Each reading that meets something becomes a",
                         "disc. Its radius is half the gap to the next",
                         "ray, so discs on one wall touch."]),
        ((-2.62, 1.98), ["Where a wall is seen at a slant the hits are",
                         "further apart. Hits on one surface are joined",
                         f"with extra discs ({joins} here, in lighter grey)."]),
        ((1.2, 2.5), ["A jump this big between neighbouring rays is",
                      "the edge of something, with open floor beside",
                      "it. It is left open: this is the way through."]),
        ((1.95, 4.45), ["The readings beyond the opening are on the",
                        "door itself."]),
    ]
    top = 44
    for index, (where, rows) in enumerate(notes, start=1):
        x, y = view.xy(*where)
        p.append(badge(x, y, index))
        p.append(badge(486, top - 4, index))
        p.append(lines(504, top, rows, leading=16, size=12.5))
        top += 16 * len(rows) + 22
    top += 6
    for fill, note in ((LIDAR, "a LiDAR reading"), (SOLID, "obstacle made from a reading"),
                       ("#9fb0bf", "obstacle added to join two readings"),
                       ("#dfe5ea", "the real wall")):
        p.append(circle(486, top - 4, 6 if fill != LIDAR else 3, fill=fill))
        p.append(text(504, top, note, size=12, fill=MUTED))
        top += 20
    seen = sum(d is not None and d <= plan.max_range_m for _, d in rays)
    return figure(820, 424, f"Seen from above: a sweep of {len(rays)} readings in a room with "
                  f"a wall and one opening. {seen} readings met something and became "
                  f"{len(obstacles)} obstacle discs; the opening stays open.", p)


def route_planning() -> str:
    """The planner's map of the room with a chair in the way, and the route found."""
    nav, plan = CFG.navigation, CFG.planner
    step = next(s for s in trace("guide")["steps"] if s["kind"] == "plan")
    obstacles = [planner.Obstacle(*o) for o in step["obstacles"]]
    goal_bearing, goal_distance = step["goal"]
    route = planner.plan(goal_bearing, goal_distance, obstacles, nav, plan)

    # The same map plan() builds, kept so that its cells can be drawn.
    target = planner.to_xy(goal_bearing, goal_distance)
    reach = max(abs(target[0]), abs(target[1]), 1.0) + 1.5
    grid = planner._Grid(half_width=reach, depth=reach, back=1.5, res=plan.resolution_m)
    for obstacle in obstacles:
        grid.stamp(obstacle, plan.clearance_m)
    grid.clear(0.0, 0.0, plan.clearance_m)
    cells, reached = planner._search(grid, target, nav.arrive_m)
    corners = planner._straighten(grid, [(0.0, 0.0)] + [grid.xy(*cell) for cell in cells[1:]])
    assert reached and len(corners) - 1 == len(route.legs)

    view = Top(212, 478, 76)
    west, east, south, north = -2.4, 2.4, -0.45, 6.0
    cell = plan.resolution_m * view.scale
    p = []
    for j in range(grid.ny):
        for i in range(grid.nx):
            x, y = grid.xy(i, j)
            if not (west <= x <= east and south <= y <= north):
                continue
            px, py = view.xy(x, y)
            extra = grid.extra[j * grid.nx + i]
            if grid.blocked[j * grid.nx + i]:
                p.append(rect(px - cell / 2, py - cell / 2, cell, cell, rx=0, fill="#a9b5c0"))
            elif extra > 0:
                p.append(rect(px - cell / 2, py - cell / 2, cell, cell, rx=0, fill=LIDAR,
                              opacity=round(0.06 + 0.16 * extra / planner.SOFT_PENALTY, 2)))
    for metre in range(-2, 3):
        x, _ = view.xy(metre, 0)
        p.append(line(x, view.xy(0, south)[1], x, view.xy(0, north)[1], stroke=LINE, width=0.5))
        p.append(text(x, view.xy(0, south)[1] + 15, f"{metre} m", "middle", 11, MUTED))
    for metre in range(0, 7):
        _, y = view.xy(0, metre)
        p.append(line(view.xy(west, 0)[0], y, view.xy(east, 0)[0], y, stroke=LINE, width=0.5))
        p.append(text(view.xy(west, 0)[0] - 6, y + 4, f"{metre} m", "end", 11, MUTED))

    for obstacle in obstacles:
        p.append(view.disc(obstacle.x, obstacle.y, obstacle.radius,
                           fill=SOLID if obstacle.label else LIDAR))
    labelled = next(o for o in obstacles if o.label)
    x, y = view.xy(0.72, labelled.y)
    p.append(text(x, y - 3, f"{labelled.label}, placed by the camera", size=12, halo=True))
    p.append(text(x, y + 14, "and the LiDAR reading on it", size=12, fill=LIDAR, halo=True))

    tx, ty = view.xy(*target)
    width = navigator.TYPICAL_SIZE_M["door"][1]
    p.append(rect(tx - width / 2 * view.scale, ty - 9, width * view.scale, 9, rx=0,
                  fill=ACCENT, stroke=INK, width=1))
    p.append(text(tx, ty - 15, "door", "middle", 12.5, weight="700"))
    p.append(circle(tx, ty, nav.arrive_m * view.scale, stroke=INK, dash="5 4"))

    for i, j in cells:
        px, py = view.xy(*grid.xy(i, j))
        p.append(circle(px, py, 2.2, fill=PATH))
    p.append(polyline([view.xy(*corner) for corner in corners], stroke=PATH, width=3.5,
                      marker="path"))
    for index, (leg, a, b) in enumerate(zip(route.legs, corners, corners[1:]), start=1):
        x, y = view.xy((a[0] + b[0]) / 2 - 0.38, (a[1] + b[1]) / 2)
        p.append(text(x, y, f"leg {index}", "end", 12.5, PATH, "700", halo=True))
    x, y = view.xy(0.2, route.clear_ahead)
    p.append(line(view.ox + 12, view.oy, view.ox + 12, y, stroke=INK, marker="ink", start="ink",
                  width=1))
    p.append(text(view.ox + 20, (view.oy + y) / 2 + 4, f"clear ahead: {route.clear_ahead:.2f} m",
                  size=12, halo=True))
    p.append(circle(view.ox, view.oy, 7, fill=PATH))
    p.append(text(view.ox - 12, view.oy + 5, "wearer", "end", 12, MUTED, halo=True))

    # The key, and what came of it.
    x, top = 440, 34
    p.append(text(x, top, "What the planner sees", size=15, weight="700"))
    top += 30
    hard = plan.clearance_m
    key = [  # (the mark, drawn level with a line of text at height y; what it means)
        (lambda y: rect(x, y - 11, 14, 14, rx=0, fill="#a9b5c0"),
         [f"Blocked: within {hard:g} m of an obstacle's edge,", "the room a walking person needs"]),
        (lambda y: rect(x, y - 11, 14, 14, rx=0, fill=LIDAR, opacity=0.2),
         [f"Free but costly: the next {planner.SOFT_MARGIN_M:g} m. The route keeps",
          "out of it where that costs little"]),
        (lambda y: circle(x + 7, y - 4, 2.5, fill=PATH),
         [f"The {len(cells)} cells the A* search chose,",
          f"each {plan.resolution_m * 100:g} cm square"]),
        (lambda y: line(x, y - 4, x + 14, y - 4, stroke=PATH, width=3.5),
         [f"The route after straightening: {len(route.legs)} legs"]),
        (lambda y: circle(x + 7, y - 4, 7, stroke=INK, dash="3 3"),
         [f"Stopping point: within {nav.arrive_m:g} m of the target"]),
    ]
    for mark, rows in key:
        p.append(mark(top))
        p.append(lines(x + 26, top, rows, leading=16, size=12.5))
        top += 16 * len(rows) + 14

    top += 12
    p.append(text(x, top, "The result", size=15, weight="700"))
    top += 26
    for index, leg in enumerate(route.legs, start=1):
        turn = side(round(leg.bearing))
        p.append(text(x, top, f"Leg {index}: {turn}, for {leg.length:.1f} m", size=12.5))
        top += 19
    first = route.legs[0]
    said = next(s["text"] for s in reversed(trace("guide")["steps"]) if s["kind"] == "speak")
    top += 8
    p.append(lines(x, top, [
        f"Leg 1 is {abs(first.bearing):.0f}° off, more than the "
        f"{nav.center_deadband_deg:g}° that counts as",
        "straight ahead, so the first action is a turn:",
    ], leading=16, size=12.5, fill=MUTED))
    p.append(text(x, top + 44, f"“{said}”", size=13.5, weight="700"))
    return figure(820, 546, "The planner's map, seen from above, for a chair between the wearer "
                  f"and a door {goal_distance:g} m away: the cells blocked around the chair, "
                  f"the cells the search chose, and the route of {len(route.legs)} legs that "
                  "goes round the chair by the left.", p)


def turn_bands() -> str:
    """The words for where something is, and for the turn towards it."""
    deadband = CFG.navigation.center_deadband_deg
    tints = ["#fde9b0", "#cfe6e3", WASH, "#d5dbe1"]
    p = []

    def dial(ox, oy, radius, title, edges, name, extra=None):
        view = Top(ox, oy, 1)
        p.append(text(ox, 30, title, "middle", 15, weight="700"))
        bands = [0.0] + edges + [180.0]
        for index, (inner, outer) in enumerate(zip(bands, bands[1:])):
            for sign in (1, -1):
                first, last = sorted((sign * inner, sign * outer))
                p.append(view.wedge(first, last, radius, fill=tints[index], stroke="#ffffff",
                                    width=1.5))
        p.append(circle(ox, oy, radius, stroke=INK, width=1.5))
        for edge in edges:
            for sign in (1, -1):
                x, y = view.polar(sign * edge, radius * 0.7)
                p.append(text(x, y + 4, f"{edge:g}°", "middle", 11.5, MUTED, halo=True))
        middles = [0.0] + [(a + b) / 2 for a, b in zip(bands[1:], bands[2:])]
        for index, middle in enumerate(middles):
            for sign in ((1,) if middle == 0 else (1, -1)):
                bearing = sign * middle
                x, y = view.polar(bearing, radius + 14)
                anchor = "middle" if middle == 0 else "start" if sign > 0 else "end"
                rows = [name(bearing)] + ([extra[index]] if extra and extra[index] else [])
                drop = -8 * (len(rows) - 1) if middle == 0 else 4
                if len(rows) > 1 and 0 < middle < 90:
                    drop -= 14  # above the dial: grow upwards, away from it
                p.append(text(x, y + drop, rows[0], anchor, 12.5, weight="700"))
                if len(rows) > 1:
                    p.append(text(x, y + drop + 16, rows[1], anchor, 11.5, MUTED))
        p.append(path(f"M{ox} {oy - 26} L{ox - 9} {oy + 8} L{ox} {oy + 2} L{ox + 9} {oy + 8} Z",
                      fill=INK))
        p.append(text(ox, oy + 30, "the wearer,", "middle", 11.5, MUTED))
        p.append(text(ox, oy + 44, "facing up", "middle", 11.5, MUTED))

    def place(bearing):
        return ENGLISH["dir." + navigator.direction_of(bearing, deadband)]

    def turn(bearing):
        asked = navigator.turn_for(bearing, deadband)
        return ENGLISH["turn." + asked] if asked else "no turn"

    expected = [None] + [f"expect about {navigator.TURN_DEGREES[size]:g}°"
                         for size in ("slight", "full", "around")]
    dial(215, 222, 104, "Where something is said to be",
         [deadband, navigator.SLIGHT_DEG, navigator.SIDE_DEG], place)
    dial(615, 222, 104, "The turn that is asked for",
         [deadband, navigator.SLIGHT_TURN_DEG, navigator.SIDE_DEG], turn, expected)
    return figure(830, 380, "Two dials seen from above. Left: the words for where something "
                  f"is, changing at {deadband:g}, {navigator.SLIGHT_DEG:g} and "
                  f"{navigator.SIDE_DEG:g} degrees. Right: the turn asked for, changing at "
                  f"{deadband:g}, {navigator.SLIGHT_TURN_DEG:g} and {navigator.SIDE_DEG:g} "
                  "degrees.", p)


def warning_distance() -> str:
    """How far ahead an obstacle is warned of, against walking speed."""
    nav, adapt = CFG.navigation, CFG.adaptation
    profile = adaptation.UserProfile()
    top_speed, top_metres = 2.0, 3.5
    left, right, bottom, top = 78, 730, 320, 34

    def at(speed: float, metres: float):
        return (left + speed / top_speed * (right - left),
                bottom - metres / top_metres * (bottom - top))

    speeds = [index * 0.02 for index in range(int(top_speed / 0.02) + 1)]
    warn = [(v, adaptation.warn_distance(nav, adapt, profile, v)) for v in speeds]
    slow = adaptation.MOVING_MPS
    urgent = [(v, v * adapt.reaction_s) for v in speeds if v >= slow]
    assert adaptation.is_urgent(urgent[-1][1] - 0.01, urgent[-1][0], adapt)

    def shape(points, **style):
        return path("M" + " L".join(f"{n(x)} {n(y)}" for x, y in points) + " Z", **style)

    p = []
    p.append(shape([at(v, d) for v, d in warn] + [at(v, d) for v, d in reversed(urgent)]
                   + [at(slow, 0), at(0, 0)], fill=ACCENT, opacity=0.28))
    p.append(shape([at(v, d) for v, d in urgent] + [at(top_speed, 0), at(slow, 0)],
                   fill=LIDAR, opacity=0.16))
    for tick in range(5):
        speed = tick * 0.5
        x, _ = at(speed, 0)
        p.append(line(x, bottom, x, top, stroke=LINE, width=0.5))
        p.append(text(x, bottom + 18, f"{speed:g}", "middle", 12, MUTED))
    for metres in range(4):
        _, y = at(0, metres)
        p.append(line(left, y, right, y, stroke=LINE, width=0.5))
        p.append(text(left - 8, y + 4, f"{metres} m", "end", 12, MUTED))
    p.append(line(left, bottom, right, bottom, stroke=INK))
    p.append(line(left, bottom, left, top, stroke=INK))
    p.append(text((left + right) / 2, bottom + 40, "Walking speed, metres a second", "middle", 12.5))
    p.append(text(left, top - 14, "Distance to what is ahead", size=12.5))

    p.append(polyline([at(v, d) for v, d in urgent], stroke=LIDAR, width=2.5))
    p.append(polyline([at(v, d) for v, d in warn], stroke=INK, width=3))
    x, y = at(0.42, 3.05)
    p.append(text(x, y, "silent", "middle", 13, MUTED))
    x, y = at(0.62, 1.32)
    p.append(text(x, y, "a warning is spoken", "middle", 13, weight="700"))
    x, y = at(1.5, 0.95)
    p.append(text(x, y, "“Stop”, cutting in", "middle", 13, LIDAR, "700"))
    p.append(text(x, y + 17, f"less than {adapt.reaction_s:g} s of walking away", "middle", 12,
                  LIDAR))
    example = 1.0
    reach = adaptation.warn_distance(nav, adapt, profile, example)
    x, y = at(example, reach)
    p.append(circle(x, y, 5, fill=INK))
    p.append(text(x - 10, y - 10, f"at {example:g} m/s: {reach:g} m", "end", 12.5, weight="700",
                  halo=True))
    x, y = at(1.66, adapt.max_warn_m)
    p.append(text(x, y - 10, f"never more than {adapt.max_warn_m:g} m", "middle", 12, halo=True))
    x, y = at(0, nav.obstacle_warn_m)
    p.append(text(x + 10, y + 20, f"standing still: {nav.obstacle_warn_m:g} m", size=12,
                  halo=True))
    return figure(770, 372, "A graph of warning distance against walking speed: "
                  f"{nav.obstacle_warn_m:g} m when standing still, rising by "
                  f"{adapt.reaction_s:g} m for each metre a second, to {adapt.max_warn_m:g} m "
                  "at most. Below a second line, an obstacle is less than "
                  f"{adapt.reaction_s:g} seconds away and the warning becomes Stop.", p)


def worked_example() -> str:
    """One recorded trip round the chair to the door, with what was said where."""
    things, blocks = ROOMS["chair"]
    run = record(things, blocks)
    stops, here, heading, trail = [], (0.0, 0.0), 0.0, [(0.0, 0.0)]
    for event in run["events"]:
        if event["type"] == "walk":
            here = tuple(event["to"])
            trail.append(here)
        elif event["type"] == "turn":
            heading = event["heading"]
        elif event["type"] == "say" and event["who"] == "assistant":
            if not stops or stops[-1][0] != here:
                stops.append((here, []))
            stops[-1][1].append(event["text"])
    assert run["reached"]

    view = Top(206, 492, 76)
    p = []
    for metre in range(0, 7):
        _, y = view.xy(0, metre)
        p.append(line(view.xy(-2.1, 0)[0], y, view.xy(2.1, 0)[0], y, stroke=LINE, width=0.5))
        p.append(text(view.xy(-2.1, 0)[0] - 6, y + 4, f"{metre} m", "end", 11, MUTED))
    for label, disc in things.items():
        target = label == "door"
        p.append(view.disc(*disc, fill=ACCENT if target else SOLID, stroke=INK if target else None,
                           width=1))
        x, y = view.xy(disc[0] + disc[2] + 0.12, disc[1])
        p.append(text(x, y + 4, label, size=12.5, weight="700"))
    p.append(polyline([view.xy(*point) for point in trail], stroke=PATH, width=3.5))
    for index, (where, _) in enumerate(stops, start=1):
        x, y = view.xy(*where)
        p.append(badge(x, y, index, fill=PATH))

    x, top = 420, 34
    p.append(text(x, top, "What was said, and where", size=15, weight="700"))
    top += 30
    for index, (where, said) in enumerate(stops, start=1):
        p.append(badge(x + 10, top - 4, index, fill=PATH))
        for sentence in said:
            p.append(text(x + 30, top, f"“{sentence}”", size=12.5))
            top += 18
        top += 12
    walked = sum(math.dist(a, b) for a, b in zip(trail, trail[1:]))
    p.append(lines(x, top + 8, [
        f"The wearer walked {walked:.1f} m and never came nearer than",
        f"{run['closest']:.2f} m to the chair. After every instruction the device",
        "looked, swept and planned again from where the wearer then stood.",
    ], leading=16, size=12, fill=MUTED))
    return figure(820, 520, "Seen from above: the path walked from the start, round the chair "
                  f"by the left, to the door, with {len(stops)} numbered places where the "
                  "assistant spoke.", p)


# ---------------------------------------------------------- charts of decisions

QUIET, ACTION = "#eef1f4", "#fde9b0"


def ladder(rows, label: str, width: int = 800, ask: int = 380, again: bool = False) -> str:
    """A chain of steps and questions, drawn from the top down.

    Each row is one of:
        ("do", text)                          something done; carry on down
        ("exit", question, answer, outcome)   on `answer` the chain ends there
        ("exit", question, answer, outcome, True)   ... and nothing is said or done
        ("either", question, if_yes, if_no)   one of two things; carry on down
        ("end", outcome)                      what is left when no exit was taken

    `again` draws an arrow from the last box back up to the first.
    """
    left, top, between, leading = 34 if again else 24, 14, 30, 16
    out_x = left + ask + 84
    out_w = width - out_x - 18
    middle = left + ask / 2

    def box(x, y, w, h, rows_, fill, weight=None, pointed=False, colour=INK):
        if pointed:
            shape = path(f"M{x + 12} {y} H{x + w - 12} L{x + w} {n(y + h / 2)} L{x + w - 12} {y + h} "
                         f"H{x + 12} L{x} {n(y + h / 2)} Z", fill=fill, stroke=INK)
        else:
            shape = rect(x, y, w, h, rx=5, fill=fill, stroke=INK)
        first = y + h / 2 - (len(rows_) - 1) * leading / 2 + 4.5
        return shape + lines(x + w / 2, first, rows_, leading=leading, anchor="middle",
                             size=12.5, weight=weight, fill=colour)

    def tall(*groups):
        return max(len(group) for group in groups) * leading + 18

    p, y, centres = [], top, []
    for index, row in enumerate(rows):
        kind = row[0]
        more = index < len(rows) - 1
        if kind in ("do", "end"):
            text_rows = wrap(row[1], 58)
            h = tall(text_rows)
            p.append(box(left, y, ask, h, text_rows, ACTION if kind == "end" else QUIET,
                         weight="700" if kind == "end" else None))
            down = None
        elif kind == "exit":
            question, answer, outcome = wrap(row[1], 54), row[2], wrap(row[3], int(out_w / 6.9))
            quiet = len(row) > 4 and row[4]
            h = tall(question, outcome)
            p.append(box(left, y, ask, h, question, "#ffffff", pointed=True))
            oh = len(outcome) * leading + 18
            p.append(line(left + ask, y + h / 2, out_x - 4, y + h / 2, stroke=INK, marker="ink"))
            p.append(text((left + ask + out_x) / 2, y + h / 2 - 7, answer, "middle", 12, MUTED, "700"))
            p.append(box(out_x, y + (h - oh) / 2, out_w, oh, outcome, QUIET if quiet else ACTION,
                         weight=None if quiet else "700", colour=MUTED if quiet else INK))
            down = "no" if answer == "yes" else "yes"
        else:  # either
            question = wrap(row[1], 54)
            choices = [wrap(row[2], int(out_w / 6.9)), wrap(row[3], int(out_w / 6.9))]
            heights = [len(choice) * leading + 18 for choice in choices]
            h = max(tall(question), sum(heights) + 10)
            p.append(box(left, y, ask, h, question, "#ffffff", pointed=True))
            at = y + (h - sum(heights) - 10) / 2
            for answer, choice, oh in zip(("yes", "no"), choices, heights):
                centre = at + oh / 2
                bend = left + ask + 30
                p.append(path(f"M{left + ask} {n(y + h / 2)} H{bend} V{n(centre)} H{out_x - 4}",
                              stroke=INK, marker="ink"))
                p.append(text(bend + 24, centre - 7, answer, "middle", 12, MUTED, "700"))
                p.append(box(out_x, at, out_w, oh, choice, QUIET))
                at += oh + 10
            down = "then"
        centres.append(y + h / 2)
        if more:
            p.append(line(middle, y + h, middle, y + h + between - 4, stroke=INK, marker="ink"))
            if down:
                p.append(text(middle + 10, y + h + between / 2 + 3, down, size=12, fill=MUTED,
                              weight="700"))
        y += h + between
    if again:
        p.append(path(f"M{left} {n(centres[-1])} H10 V{n(centres[0])} H{left - 4}", stroke=INK,
                      marker="ink"))
    return figure(width, int(y - between + top), label, p)


def wrap(sentence: str, chars: int) -> list:
    import textwrap

    return textwrap.wrap(sentence, chars) or [""]


def main_loop() -> str:
    """One pass of Assistant.step()."""
    faces = CFG.faces.every_n_frames
    return ladder([
        ("do", "Act on every phrase waiting in the queue of recognised speech"),
        ("exit", "Is the wearer being guided along a planned route, and is the time "
                 "allowed for the last instruction up?", "yes",
         "Run one guidance cycle (section 8). The step ends here"),
        ("exit", "Read a picture from the camera. Was there one?", "no",
         "Wait 0.1 s. The step ends here", True),
        ("do", "Run the detector on the picture. Read the LiDAR"),
        ("do", "Update the walking speed. Notice a pause. Notice a crowd"),
        ("either", "Is the wearer being guided?",
         "Check for arrival and for danger. With no route planner, steer by the camera",
         "Warn of an obstacle ahead. Describe the view if the wearer has just paused"),
        ("do", f"Every {faces} pictures: name the registered people in view"),
    ], "One pass of the main loop, from the top: act on commands; run a guidance cycle if one "
       "is due; otherwise take a picture, detect, read the LiDAR, track movement, then either "
       "watch the guidance or warn of obstacles, and lastly name the people in view.")


def next_action() -> str:
    """How next_instruction() picks the one action to speak."""
    nav = CFG.navigation
    band = nav.center_deadband_deg
    return ladder([
        ("exit", "Has the planner found any route at all?", "no", "Blocked"),
        ("exit", f"Is the first leg more than {band:g}° off? "
                 f"({band * navigator.AFTER_TURN_SLACK:g}° if the wearer has just turned)", "yes",
         "Turn towards it"),
        ("do", "Count the steps to walk: the least of what the leg calls for, "
               f"{navigator.STEP_MARGIN:.0%} of the distance that is clear ahead, "
               f"{navigator.MAX_STEPS} steps, and as far as stays within "
               f"{navigator.MAX_DRIFT_M:g} m of the leg"),
        ("exit", "Is that at least one step?", "yes",
         "Move forward that many steps. Name the next turn too, if the whole leg is walked "
         "and another follows"),
        ("exit", "Is the leg too short to walk, with another after it?", "yes",
         "The turn that follows"),
        ("exit", "Does the leg end at the target, no more than one step away?", "yes",
         "Arrived"),
        ("exit", f"Does the leg lean {navigator.NUDGE_DEG:g}° or more to one side?", "yes",
         "Turn slightly to that side"),
        ("end", "Otherwise: blocked"),
    ], "The chain of questions that picks the next action: no route means blocked; a first leg "
       "off to one side means a turn; otherwise as many steps as are safe; and when not even "
       "one step fits, the following turn, arrival, a slight turn away, or blocked.")


def warning_decision() -> str:
    """How _warn_obstacle() decides whether to speak."""
    adapt = CFG.adaptation
    return ladder([
        ("exit", "Is the LiDAR reading inside the warning distance?", "no", "Say nothing", True),
        ("exit", "Is the thing ahead what the wearer is being guided to?", "yes",
         "Say nothing", True),
        ("exit", f"Is it less than {adapt.reaction_s:g} s of walking away?", "yes",
         "“Stop, chair ahead, 1.1 meters.” It cuts in on whatever is being said, but is not "
         "repeated within 1.5 s"),
        ("exit", "Is the assistant still speaking, or was the last warning too recent?", "yes",
         "Say nothing", True),
        ("exit", "Is it the same thing at the same distance, already warned of in the last "
                 "15 s?", "yes", "Say nothing", True),
        ("end", "Otherwise: “Chair detected at 0.9 meters.”"),
    ], "The chain of questions before an obstacle warning: nothing is said if the reading is "
       "outside the warning distance or is the guidance target; an obstacle under 1.5 seconds "
       "away gets Stop; otherwise a warning is spoken unless the assistant is busy or has just "
       "warned of the same thing.")


def guidance_cycle() -> str:
    """One pass of _plan_route()."""
    nav, plan = CFG.navigation, CFG.planner
    readings = int(2 * plan.sweep_half_angle_deg / plan.sweep_step_deg) + 1
    views = len(CFG.servo.scan_angles)
    return ladder([
        ("do", "Look where the target should be by now. If it is not there, scan all "
               f"{views} angles for it"),
        ("exit", "Was the target seen?", "no",
         f"Say it is being looked for, and look again. After {nav.lost_frames} looks without "
         "it: “I lost sight of the door.” Guidance ends"),
        ("exit", "Is it within reach?", "yes", "“You have reached the door.” Guidance ends"),
        ("do", f"Sweep the LiDAR across the room: {readings} readings"),
        ("do", "Gather the obstacles: from the sweep, from what the camera named, and from "
               "what is remembered"),
        ("do", "Search for the shortest safe route (A*), and choose the next single action"),
        ("exit", f"Stuck? Blocked {nav.lost_frames} times running, turned through a full "
                 "circle without a walk, or ten actions with the target no nearer", "yes",
         "“I cannot find a clear way to the door.” Guidance ends"),
        ("end", "Say the action, wait for the wearer to do it, and begin again"),
    ], "One guidance cycle from the top: find the target again; stop if it is lost or within "
       "reach; otherwise sweep the LiDAR, gather the obstacles, plan the route and choose one "
       "action; give up aloud if stuck; otherwise say the action, wait, and begin again.",
        again=True)


def speech_out() -> str:
    """Which voice speaks a sentence."""
    from visual_assist import speech

    return ladder([
        ("do", "say(sentence) puts it on the queue. The speaker thread takes the next one"),
        ("exit", "Is Google's voice wanted? (backend gtts, or backend auto with a language "
                 "other than English)", "no",
         "The offline voice speaks it: espeak-ng on the Pi, the Windows voice on a PC"),
        ("exit", "Fetch the sentence from Google as an MP3, unless it is already kept, and "
                 "play it. Did that work?", "yes", "Spoken"),
        ("end", "Otherwise the offline voice speaks it, and Google's voice is left alone for "
                f"{speech.RETRY_AFTER_S:g} s"),
    ], "How a sentence is spoken: it waits in a queue; the offline voice speaks it unless "
       "Google's voice is wanted; if Google's voice fails, the offline voice speaks it "
       "instead.")


# ------------------------------------------------------------ charts of structure

def guidance_states() -> str:
    """The states guidance moves between."""
    p = []

    def state(x, y, w, name):
        p.append(rect(x, y, w, 50, rx=10, fill=QUIET, stroke=INK))
        p.append(text(x + w / 2, y + 30, name, "middle", 13.5, weight="700"))

    def move(x1, y1, x2, y2, note, nx, ny, anchor="middle"):
        p.append(line(x1, y1, x2, y2, stroke=INK, marker="ink"))
        p.append(text(nx, ny, note, anchor, 12, MUTED, halo=True))

    state(44, 34, 160, "Not guiding")
    state(556, 34, 190, "Finding the target")
    state(540, 184, 222, "Planning the next action")
    state(44, 184, 236, "Waiting for the wearer to act")
    p.append(circle(16, 59, 6, fill=INK))
    p.append(line(22, 59, 40, 59, stroke=INK, marker="ink"))
    move(204, 50, 552, 50, "“Guide me to the door”", 380, 42)
    move(556, 68, 208, 68, "not found, or already within reach", 380, 86)
    move(651, 84, 651, 180, "found and measured", 661, 136, "start")
    move(540, 200, 284, 200, "the action is spoken", 412, 192)
    move(280, 218, 536, 218, "time is up, the wearer paused, or a warning was given", 412, 252)
    move(100, 184, 100, 88, "arrived, or the wearer said “Stop”", 110, 158, "start")
    p.append(path("M600 184 V122 H172 V88", stroke=INK, marker="ink"))
    p.append(text(386, 115, "arrived, target lost, or no way found", "middle", 12, MUTED,
                  halo=True))
    return figure(820, 264, "Four states. Not guiding leads to finding the target when the "
                  "wearer asks to be guided. Once found, the assistant alternates between "
                  "planning the next action and waiting for the wearer to act, until arrival, "
                  "loss of the target, no way through, or Stop.", p)


def threads() -> str:
    """The three threads and the two queues between them."""
    p = []

    def lane(x, title, rows):
        p.append(rect(x, 52, 196, 150, rx=6, fill=QUIET, stroke=INK))
        p.append(text(x + 98, 78, title, "middle", 14.5, weight="700"))
        p.append(lines(x + 98, 106, rows, leading=17, anchor="middle", size=12.5))

    def queue(x, title):
        for slot in range(4):
            p.append(rect(x + slot * 17, 112, 15, 30, rx=2, fill="#ffffff", stroke=INK, width=1.25))
        p.append(lines(x + 33, 160, title, leading=14, anchor="middle", size=11.5, fill=MUTED))

    lane(20, "Listener thread", ["Listens on the", "microphone and turns", "speech into text"])
    lane(312, "Main thread", ["Assistant.step():", "commands, camera,", "detector, LiDAR,",
                              "servos, decisions"])
    lane(604, "Speaker thread", ["Speaks one", "sentence at a time"])
    queue(231, ["queue of", "phrases"])
    queue(523, ["queue of", "sentences"])
    for x1, x2 in ((216, 229), (299, 310), (508, 521), (591, 602)):
        p.append(line(x1, 127, x2, 127, stroke=INK, marker="ink", width=1.5))
    p.append(path("M410 52 C410 14, 702 14, 702 48", stroke=LIDAR, dash="6 4", marker="lidar"))
    p.append(text(556, 20, "urgent: empty the queue and cut off the sentence being spoken",
                  "middle", 12, LIDAR, halo=True))
    p.append(path("M702 202 C702 262, 118 262, 118 206", stroke=MUTED, dash="6 4", marker="muted"))
    p.append(text(410, 252, "while a sentence is being spoken, wait before listening", "middle",
                  12, MUTED, halo=True))
    return figure(820, 276, "Three threads side by side. The listener thread puts recognised "
                  "phrases on a queue that the main thread empties at the start of every step. "
                  "The main thread puts sentences on a queue that the speaker thread speaks one "
                  "at a time.", p)


def training_pipeline() -> str:
    """From the public photo collections to the models in use."""
    p = []

    def box(x, y, w, rows, fill="#ffffff", weight=None):
        h = len(rows) * 16 + 16
        p.append(rect(x, y, w, h, rx=5, fill=fill, stroke=INK))
        p.append(lines(x + w / 2, y + 21, rows, leading=16, anchor="middle", size=12.5,
                       weight=weight))
        return y + h / 2

    def arrow(x1, y1, x2, y2, note=None, above=True):
        p.append(line(x1, y1, x2 - 4, y2, stroke=INK, marker="ink"))
        if note:
            p.append(text((x1 + x2) / 2, (y1 + y2) / 2 + (-7 if above else 16), note, "middle",
                          11.5, MUTED, family=MONO, halo=True))

    for x, title in ((20, "Photo collections"), (290, "Datasets"), (604, "Trained models")):
        p.append(text(x, 26, title, size=13, fill=MUTED, weight="700"))
    door_src = box(20, 44, 170, ["DoorDetect"])
    open_src = box(20, 104, 170, ["Open Images"])
    note_src = box(20, 164, 170, ["Mendeley banknotes"])
    doors = box(290, 58, 170, ["Doors", "1,715 + 229 photos"], QUIET)
    notes = box(290, 150, 170, ["Banknotes", "2,575 + 625 photos"], QUIET)
    door_model = box(604, 58, 196, ["weights/doors.pt"], ACTION, "700")
    note_model = box(604, 150, 196, ["weights/currency.pt"], ACTION, "700")
    ready = box(604, 226, 196, ["weights/yolov5nu.pt", "ready-made: 80 objects"],
                ACTION)

    arrow(190, door_src, 290, doors - 8)
    arrow(190, open_src, 290, doors + 8)
    p.append(text(198, 50, "prepare_doors.py", "start", 11.5, MUTED, family=MONO, halo=True))
    arrow(190, open_src + 6, 290, notes - 8)
    arrow(190, note_src, 290, notes + 8)
    p.append(text(198, 214, "prepare_currency.py", "start", 11.5, MUTED, family=MONO,
                  halo=True))
    p.append(line(375, 106, 375, 146, stroke=INK, marker="ink"))
    p.append(text(383, 130, "indoor photos, for “no note”", size=11.5, fill=MUTED, halo=True))
    arrow(460, doors, 604, door_model, "train.py doors")
    arrow(460, notes, 604, note_model, "train.py currency")

    # Everything on the right is used twice: on the device, and converted for the page.
    p.append(path(f"M800 {n(door_model)} H812 V352 M800 {n(note_model)} H812 M800 {n(ready)} H812",
                  stroke=INK))
    device = box(520, 300, 250, ["On the device:", "detector.py and currency.py"], QUIET)
    export = box(520, 366, 250, ["export_models.py:", "the same three, as ONNX files"], QUIET)
    p.append(path(f"M812 {n(device)} H774 M812 352 V{n(export)} H774", stroke=INK, marker="ink"))
    p.append(line(812, device, 774, device, stroke=INK, marker="ink"))
    page = box(250, 374, 190, ["The demo page"], ACTION, "700")
    arrow(520, export, 440, page)
    p.append(text(480, export - 8, "demo/models", "middle", 11.5, MUTED, family=MONO, halo=True))
    return figure(830, 430, "Two photo collections become the door dataset and, with a third, "
                  "the banknote dataset. Training turns each into a model file. Those two and a "
                  "ready-made model are loaded on the device, and converted to ONNX for the "
                  "demo page.", p)


def demo_architecture() -> str:
    """What runs where when someone opens the demo page."""
    p = []

    def box(x, y, w, h, rows, fill="#ffffff", weight=None):
        p.append(rect(x, y, w, h, rx=5, fill=fill, stroke=INK))
        first = y + h / 2 - (len(rows) - 1) * 8 + 4.5
        p.append(lines(x + w / 2, first, rows, leading=16, anchor="middle", size=12.5,
                       weight=weight))

    def group(x, y, w, h, title):
        p.append(rect(x, y, w, h, rx=6, fill=QUIET, stroke=MUTED, width=1, dash="5 4"))
        p.append(text(x + w / 2, y + 20, title, "middle", 12, MUTED, "700"))

    def pair(x1, x2, y, forward, back):
        """Two arrows between neighbours: one each way, each with its name."""
        p.append(line(x1, y - 9, x2 - 4, y - 9, stroke=INK, marker="ink"))
        p.append(line(x2, y + 9, x1 + 4, y + 9, stroke=INK, marker="ink"))
        p.append(text((x1 + x2) / 2, y - 15, forward, "middle", 11, MUTED, halo=True))
        p.append(text((x1 + x2) / 2, y + 24, back, "middle", 11, MUTED, halo=True))

    box(20, 16, 790, 40, ["GitHub Pages: static files only. Nothing is sent back to it"], ACTION,
        "700")
    p.append(rect(20, 112, 790, 384, rx=8, stroke=INK, width=1.5))
    p.append(text(36, 134, "The visitor's browser", size=13, weight="700"))

    group(36, 150, 190, 216, "ONNX Runtime Web")
    box(52, 244, 158, 72, ["yolov5nu.onnx", "doors.onnx", "currency.onnx"])
    group(290, 150, 200, 330, "The page's scripts")
    box(306, 184, 168, 52, ["demo.js", "the simulated room"])
    box(306, 254, 168, 52, ["live.js", "the camera section"])
    box(306, 398, 168, 52, ["device.js", "the device section"])
    group(604, 150, 192, 330, "Pyodide: Python")
    box(619, 184, 162, 52, ["simulation.record()"])
    box(619, 254, 162, 52, ["live.LiveAssistant"])
    box(619, 362, 162, 96, ["the project's modules:", "assistant, planner,", "navigator, phrases,",
                            "adaptation, commands"])
    box(52, 398, 158, 52, ["The visitor's", "camera"], ACTION)

    for x, note in ((131, "demo/models"), (390, "recordings.json, device.json"),
                    (700, "visual_assist/ files")):
        p.append(line(x, 56, x, 146, stroke=INK, marker="ink"))
        p.append(text(x, 88, note, "middle", 11.5, MUTED, family=MONO, halo=True))
    pair(474, 619, 210, "room, language", "events to replay")
    pair(474, 619, 280, "detections, questions", "sentences")
    pair(306, 210, 280, "picture", "boxes, scores")
    p.append(path("M210 424 H334 V310", stroke=INK, marker="ink"))
    p.append(text(258, 416, "pictures", "middle", 11, MUTED, halo=True))
    p.append(line(404, 306, 404, 394, stroke=INK, marker="ink"))
    p.append(line(444, 398, 444, 310, stroke=INK, marker="ink"))
    p.append(text(398, 354, "answer", "end", 11, MUTED, halo=True))
    p.append(text(450, 354, "ask", "start", 11, MUTED, halo=True))
    p.append(line(700, 306, 700, 362, stroke=MUTED, dash="4 3"))
    p.append(text(708, 338, "both run on", size=11, fill=MUTED, halo=True))
    return figure(820, 512, "The demo page's parts. GitHub Pages only serves files. In the "
                  "visitor's browser, the page's three scripts sit between ONNX Runtime Web, "
                  "which runs the three models on the camera's pictures, and Pyodide, which "
                  "runs the project's own Python.", p)


FIGURES = {
    "threads": threads,
    "guidance-states": guidance_states,
    "main-loop": main_loop,
    "guidance-cycle": guidance_cycle,
    "next-action": next_action,
    "warning-decision": warning_decision,
    "speech-out": speech_out,
    "training-pipeline": training_pipeline,
    "demo-architecture": demo_architecture,
    "wiring": wiring,
    "lidar-frame": lidar_frame,
    "camera-geometry": camera_geometry,
    "beam-blocked": beam_blocked,
    "scan-and-sweep": scan_and_sweep,
    "readings-to-obstacles": readings_to_obstacles,
    "route-planning": route_planning,
    "turn-bands": turn_bands,
    "warning-distance": warning_distance,
    "worked-example": worked_example,
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        target = OUT / f"{name}.svg"
        target.write_text(draw(), encoding="utf-8", newline="\n")
        print(f"wrote {target.relative_to(HERE.parent)} ({target.stat().st_size // 1024} KB)")

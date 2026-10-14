# fishgen

Procedural pixel-art fish generator. Every seed gives a different fish (body shape, fins,
tail, palette and pattern) with four animations that hand off seamlessly into each other:

```
swim_right ──▶ turn_right_to_left ──▶ swim_left ──▶ turn_left_to_right ──▶ swim_right
   ↻ loop                                ↻ loop
```

The turn is a 3/4 rotation: the fish swings its head towards the viewer, shows its face,
and comes out facing the other way, with its body curving behind its head.

## Quick start: the designer window

```sh
py gui.py          # Windows   (python3 gui.py elsewhere)
```

That's it. The first run creates `.venv` and installs numpy + Pillow by itself.

Pick a **species** (Goldfish, Clownfish, Angelfish, Tuna, Eel, Pufferfish, Betta, Neon tetra,
Koi, Discus, Mackerel, Piranha, or Random with a chosen body type), then tweak anything on the
tabs - body shape, eye, every fin, colours and pattern, sprite size and frame counts. The fish
re-renders live on the right. The **Export** tab writes the same files as `fishgen generate`
(named after the species and seed, e.g. `out/koi_5/koi_5.png`).

## Install (command line)

```sh
python -m venv .venv
.venv/Scripts/activate            # Windows  (source .venv/bin/activate elsewhere)
pip install -e ".[preview,dev]"
```

## Usage

```sh
fishgen                                         # opens the designer window
fishgen generate --seed 42                      # all formats into ./out/fish_42/
fishgen generate --species Clownfish --count 5  # five different clownfish
fishgen generate --count 20 --size 48 --out fish    # 20 fish, 48x48 sprites
fishgen generate --seed 7 --size 64x32 --formats sheet,godot
fishgen generate --seed 7 --formats gif --scale 8 --bg "#1d3557"
fishgen preview --fish 5                        # live aquarium window
```

`python -m fishgen ...` works too.

### Options

| option | default | |
|---|---|---|
| `--seed N` | random | same seed = same fish |
| `--species NAME` | random | preset fish type (also works with `preview` and `gui`) |
| `--size N` or `WxH` | `32` | sprite size in pixels |
| `--swim-frames` | `8` | frames per swim cycle |
| `--turn-frames` | `10` | frames per turn-around |
| `--fps` | `10` | playback rate written into every format |
| `--pitch` | `0.25` | camera tilt (radians); higher shows more of the back |
| `--formats` | all | any of `gif,sheet,godot,frames` |
| `--count` | `1` | batch: consecutive seeds, one folder each |
| `--scale` | `4` | GIF upscale factor |
| `--frame-scale` | `1` | upscale for loose PNG frames |
| `--bg #RRGGBB` | transparent | opaque GIF background |
| `--padding` | `0` | gap between sprite-sheet cells |
| `--godot-dir` | `res://` | folder the `.tres` expects the PNG in |

### Preview keys

`SPACE` new fish · `←/→` previous/next seed · `T` turn · `G` show all animation frames ·
`P` pause · `+/-` zoom · `E` export the fish on screen · `ESC` quit

## Output

```
out/fish_42/
  fish_42.png          sprite sheet: one row per animation
  fish_42.json         frame rects, fps, loop flags, `next` animation, palette, genome
  fish_42.tres         Godot 4 SpriteFrames (uses fish_42.png)
  gif/                 one GIF per animation + showcase.gif (the whole cycle)
  frames/<anim>/       individual PNG frames
```

All frames share the same canvas and the fish is always centred, so the sprite's pivot is
the middle of the frame (`meta.pivot` in the JSON).

### Playing it in a game

Non-looping animations (the turns) name the animation that follows them in `next`. Only
start a turn when a swim loop **finishes** - that is the frame the turn was built to
continue from, so the tail beat never skips. To move the fish, scale its speed by the
facing of the current frame (the preview uses `cos(yaw)`), so it slows into the turn and
speeds out of it.

Godot 4 example (`AnimatedSprite2D` with the `.tres` assigned):

```gdscript
extends AnimatedSprite2D

var want_turn := false

func _ready():
    play("swim_right")
    animation_looped.connect(_on_loop)
    animation_finished.connect(_on_finished)

func _on_loop():
    if want_turn and animation.begins_with("swim"):
        want_turn = false
        play("turn_right_to_left" if animation == "swim_right" else "turn_left_to_right")

func _on_finished():  # turns don't loop
    play("swim_left" if animation == "turn_right_to_left" else "swim_right")
```

## How it works

1. **Genome** (`genome.py`) - a seeded RNG picks an archetype (slender, torpedo, classic,
   deep, round) and then every proportion, fin, tail shape, colour and pattern.
   Species presets (`species.py`) fix the archetype and override some genes; the seed varies the rest.
2. **Model** (`geometry.py`) - the body is a 3D tube lofted along a spine; fins are flat
   meshes attached to it. A pose bends the spine (travelling swim wave + tail flex + a
   C-curve while turning) and rotates it around the vertical axis.
3. **Rasterizer** (`raster.py`) - z-buffered triangles sampled once at each pixel centre,
   so the result is crisp pixel art rather than a blurry downscale.
4. **Shading** (`render.py`) - lighting is quantised to a limited, hue-shifted palette
   (`palette.py`), patterns are evaluated on the body surface so they turn with the fish,
   then eyes, fin edges and a 1px outline are drawn and stray pixels cleaned up.
5. **Animation** (`animation.py`) - swim and turn poses share a continuous tail-beat phase
   so every hand-off is seamless.

## Tests

```sh
pytest
```

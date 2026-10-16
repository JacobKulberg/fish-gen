<div align="center">

# 🐟 fishgen

**Procedural pixel-art sea creatures, with game-ready swim and turn-around animations.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Godot 4](https://img.shields.io/badge/exports-Godot%204-478CBF?logo=godotengine&logoColor=white)](#playing-it-in-a-game)
![38 species](https://img.shields.io/badge/species-38-orange)

<img src="docs/images/hero.gif" alt="Twelve generated animals swimming and turning around" width="636">

</div>

Every seed gives a different fish (body shape, fins, tail, palette and pattern) with four
animations that hand off seamlessly into each other. Besides fish it makes sharks, whales
and dolphins, rays, squid and octopus, and sea urchins. Output is crisp pixel art at any
sprite size, exported as GIFs, a sprite sheet + JSON, loose PNG frames, or a Godot 4
`SpriteFrames` resource.

- 🎲 **Infinite variety** - same seed, same fish; a new seed, a new fish
- 🦈 **7 kinds, 38 species presets** - from clownfish to hammerheads, mantas and octopus
- 🔄 **Seamless 3/4 turn-arounds** - the fish shows its face mid-turn, no sprite flipping
- 🖌️ **Designer GUI** - tweak every gene and watch it re-render live
- 🎮 **Game-ready exports** - sprite sheets with metadata and drop-in Godot 4 resources

<table>
<tr>
<td align="center"><img src="docs/images/showcase.gif" alt="A koi swimming and turning around" width="360"><br><sub>one koi, full cycle</sub></td>
<td align="center"><img src="docs/images/turn.gif" alt="A great white shark turning around" width="360"><br><sub>a great white turning around</sub></td>
</tr>
</table>

```
swim_right ──▶ turn_right_to_left ──▶ swim_left ──▶ turn_left_to_right ──▶ swim_right
   ↻ loop                                ↻ loop
```

The turn is a 3/4 rotation: the fish swings its head towards the viewer, shows its face,
and comes out facing the other way, with its body curving behind its head.

Pufferfish get two more, `puff_right` and `puff_left`: the slim little fish swells into a
spiky ball, holds it, and deflates back into its swim (both hand back to the matching swim loop).

**Sizes.** Every animal has a size, and the sprite canvas scales with it: `--size` (default 32)
is the canvas for a medium-sized fish, so a neon tetra comes out around 14px, a great white
around 70px and a humpback around 90px - put them in one tank and the sharks are bigger than
the fish. Pass `--fixed-size` (or untick *Scale canvas with animal size* in the designer) to
render every animal at exactly `--size`.

## Gallery

**24 random fish** (`fishgen generate --count 24`): no two alike.

<img src="docs/images/random_fish.gif" alt="A grid of 24 randomly generated fish swimming" width="100%">

**Every species preset**, seed 1 (row by row: fish, sharks, whales & dolphins, rays, squid & cuttlefish, octopus, urchins):

<img src="docs/images/species.png" alt="All 38 species presets" width="100%">

## Quick start: the designer window

```sh
git clone https://github.com/JacobKulberg/fish-gen.git
cd fish-gen
py gui.py          # Windows   (python3 gui.py elsewhere)
```

That's it. The first run creates `.venv` and installs numpy + Pillow by itself.

Pick a **kind** of animal first, then a **species** of that kind (or *Random* for a random
animal of that kind), then tweak anything on the tabs - body shape, eye, fins, wings and arms,
colours and pattern, sprite size and frame counts. Only the controls that apply to the chosen
kind are shown. The animal re-renders live on the right.

The kind is always decided first and every gene is then drawn to suit it, so a random shark
is always a shark (shark tail, gill slits, two dorsal fins, stiff side fins) and only its
proportions, snout, colours and markings vary; a random ray is always a flat winged ray, and so on.
Presets work the same way: the species fixes what makes it that species (a Tiger shark's
stripes, an Orca's markings) and each seed is a different individual of it, with its own
build, fin, wing or arm sizes, shading and the odd scar.

| kind | species |
|---|---|
| Fish | Random (with a chosen body type), Goldfish, Clownfish, Angelfish, Tuna, Eel, Pufferfish, Betta, Neon tetra, Koi, Discus, Mackerel, Piranha |
| Shark | Tiger shark, Great white shark, Hammerhead shark, Whale shark, Mako shark, Blacktip reef shark |
| Whale & dolphin | Orca, Bottlenose dolphin, Common dolphin, Beluga, Humpback whale |
| Ray | Cownose ray, Manta ray, Spotted eagle ray, Stingray, Bluespotted stingray |
| Squid & cuttlefish | Squid, Giant squid, Cuttlefish |
| Octopus | Octopus, Blue-ringed octopus, Dumbo octopus |
| Sea urchin | Purple sea urchin, Long-spined urchin, Red pencil urchin, Green sea urchin |

Each animal moves its own way: whales and dolphins beat their flukes up and down, rays flap
(or ripple) their wings and are seen a little from above, squid and octopus jet mantle-first
with their arms trailing and flaring, and urchins sit still while their spines sway. The **Export** tab writes the same files as `fishgen generate`
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
fishgen generate --species "Tiger shark" --size 48
fishgen generate --species Pufferfish           # includes the puff-up animations
fishgen generate --kind ray --count 4           # four random rays: ./out/ray_<seed>/
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
| `--kind KIND` | fish | `fish`, `shark`, `whale`, `ray`, `squid`, `octopus` or `urchin`: a random animal of that kind |
| `--species NAME` | random | preset (implies its kind; also works with `preview` and `gui`) |
| `--size N` or `WxH` | `32` | sprite size in pixels for a medium-sized animal (bigger animals get bigger sprites) |
| `--fixed-size` | off | use exactly `--size` for every animal |
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

<img src="docs/images/sheet.png" alt="A clownfish sprite sheet: one row per animation" width="100%">

```
out/fish_42/
  fish_42.png          sprite sheet: one row per animation
  fish_42.json         frame rects, fps, loop flags, `next` animation, palette, genome
  fish_42.tres         Godot 4 SpriteFrames (uses fish_42.png)
  gif/                 one GIF per animation + showcase.gif (the whole cycle)
  frames/<anim>/       individual PNG frames
```

All frames of one animal share the same canvas and the animal is always centred, so the
sprite's pivot is the middle of the frame (`meta.pivot` in the JSON). Canvas sizes differ
between animals unless you pass `--fixed-size`.

### Playing it in a game

Non-looping animations (the turns, and a pufferfish's puffs) name the animation that follows
them in `next`. Only start a turn or a puff when a swim loop **finishes** - that is the frame the turn was built to
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

func _on_finished():  # turns and puffs don't loop
    play("swim_left" if animation in ["turn_right_to_left", "puff_left"] else "swim_right")
```

## How it works

1. **Genome** (`genome.py`, `kinds.py`) - the animal's kind comes first. For a fish, a seeded
   RNG picks an archetype (slender, torpedo, classic, deep, round) and then every proportion,
   fin, tail shape, colour and pattern. Every other kind has its own generator that sets all of
   its genes within that kind's ranges (some kinds have variants, e.g. requiem shark or
   hammerhead, eagle ray or stingray). Species presets (`species.py`) belong to one kind, ask
   its generator for a body and override the genes that make the species; the seed varies the rest.
2. **Model** (`geometry.py`) - the body is a 3D tube lofted along a spine; fins are flat
   meshes attached to it. A pose bends the spine (travelling swim wave + tail flex + a
   C-curve while turning) and rotates it around the vertical axis.
   Other animals reuse the same spine: flukes lie flat and whales bend it vertically, rays and
   squid add flapping wing sheets, cephalopod arms are tapering tubes hung off the end of the
   body, a hammerhead's head is a flat sheet with the eyes on its tips, and urchin spines are
   line segments drawn depth-tested after the mesh.
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

## License

[MIT](LICENSE). Generated sprites are yours to use however you like, including in commercial games.

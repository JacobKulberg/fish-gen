import json
import math

import numpy as np
import pytest

from fishgen import FishGenome, generate
from fishgen.animation import SHOWCASE_ORDER, build_animations, smooth_timing
from fishgen.cli import main, parse_size


def opaque(frame: np.ndarray) -> np.ndarray:
    return frame[..., 3] > 0


@pytest.fixture(scope="module")
def sprite():
    return generate(42, 32, 32, scale_by_size=False)


def test_same_seed_same_fish():
    a, b = generate(7, 24, 24, scale_by_size=False), generate(7, 24, 24, scale_by_size=False)
    for name in a.animations:
        for fa, fb in zip(a.animations[name].frames, b.animations[name].frames):
            assert np.array_equal(fa, fb)


def test_different_seeds_differ():
    assert FishGenome.random(1) != FishGenome.random(2)


def test_frames_have_expected_shape(sprite):
    for anim in sprite.animations.values():
        assert anim.frames
        for f in anim.frames:
            assert f.shape == (32, 32, 4)
            assert f.dtype == np.uint8


def test_non_square_size():
    s = generate(3, 48, 32, scale_by_size=False)
    assert s.animations["swim_right"].frames[0].shape == (32, 48, 4)


def test_fish_fits_inside_canvas(sprite):
    # Every frame is drawn, and nothing touches the canvas edge.
    for anim in sprite.animations.values():
        for f in anim.frames:
            a = f[..., 3]
            assert (a > 0).sum() > 20
            assert not a[0].any() and not a[-1].any()
            assert not a[:, 0].any() and not a[:, -1].any()


def test_frames_are_pixel_art():
    """By default: hard edges, one small palette shared by every frame, and a dark outline."""
    s = generate(3, 96, 64, scale_by_size=False)
    assert s.style == "pixel" and s.palette is not None and len(s.palette) <= 32
    pal = {tuple(int(v) for v in c) for c in s.palette}
    for anim in s.animations.values():
        for f in anim.frames:
            assert set(np.unique(f[..., 3])) <= {0, 255}  # no half-transparent pixels
            assert {tuple(c) for c in f[opaque(f)][:, :3]} <= pal
    # The pixels round the silhouette are the outline: darker than the animal inside it.
    f = s.animations["swim_right"].frames[0]
    on = opaque(f)
    inner = on.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        inner &= np.roll(on, (dy, dx), axis=(0, 1))
    lum = f[..., :3].astype(float).mean(-1)
    assert lum[on & ~inner].mean() < 0.6 * lum[inner].mean()


@pytest.mark.parametrize("size", [16, 24, 32, 48, 64, 96, 128])
def test_pixel_art_sizes(size):
    from fishgen.species import make_genome

    s = generate(1, size, size, genome=make_genome("Clownfish", 1), scale_by_size=False, swim_frames=2,
                 turn_frames=2)
    f = s.animations["swim_right"].frames[0]
    assert f.shape == (size, size, 4) and opaque(f).sum() > size
    assert len(s.palette) <= 32


def test_palette_size_and_outline_options():
    s = generate(5, 48, 48, scale_by_size=False, colors=6, outline=False, swim_frames=2, turn_frames=2)
    assert len(s.palette) == 6
    t = generate(5, 48, 48, scale_by_size=False, colors=6, swim_frames=2, turn_frames=2)
    assert 6 < len(t.palette) <= 9  # plus a few outline shades
    # The outline goes round the outside, so the outlined sprite covers more pixels.
    f0, f1 = s.animations["swim_right"].frames[0], t.animations["swim_right"].frames[0]
    assert opaque(f1).sum() > opaque(f0).sum()


def test_smooth_style_is_full_colour_and_anti_aliased():
    s = generate(3, 96, 64, scale_by_size=False, style="smooth")
    assert s.palette is None
    f = s.animations["swim_right"].frames[0]
    colours = {tuple(c) for c in f[opaque(f)][:, :3]}
    assert len(colours) > 200  # shaded, textured, not a handful of palette slots
    edge = f[..., 3]
    assert ((edge > 0) & (edge < 255)).sum() > 20  # soft, supersampled silhouette
    with pytest.raises(ValueError):
        generate(3, 32, 32, scale_by_size=False, style="watercolour")


def test_native_code_matches_numpy(monkeypatch):
    """The C hot loops give the same frames as the numpy fallback."""
    from fishgen import native
    from fishgen.render import FishRenderer
    from fishgen.species import make_genome

    if native.lib is None:
        pytest.skip("no C compiler: running on the numpy fallback")
    poses = build_animations(2, 2)["swim_right"].poses
    for name in ("Great white shark", "Long-spined urchin", "Octopus"):
        r = FishRenderer(make_genome(name, 1), 48, 48, fit_poses=poses)
        fast = r.render(poses[1])
        monkeypatch.setattr(native, "lib", None)
        slow = r.render(poses[1])
        monkeypatch.undo()
        assert np.abs(fast.astype(int) - slow.astype(int)).max() <= 1, name


def test_animation_chain_is_closed():
    anims = build_animations()
    for name, anim in anims.items():
        assert anim.next in anims
    assert anims["turn_right_to_left"].next == "swim_left"
    assert anims["turn_left_to_right"].next == "swim_right"


def test_turns_hand_off_smoothly():
    """Phase/yaw continue across every hand-off with no bigger jump than inside a loop."""
    anims = build_animations(swim_frames=8, turn_frames=10)

    def phase_step(p0, p1):
        return (p1.phase - p0.phase) % 1.0

    for a, b in [("swim_right", "turn_right_to_left"), ("turn_right_to_left", "swim_left"),
                 ("swim_left", "turn_left_to_right"), ("turn_left_to_right", "swim_right")]:
        last, first = anims[a].poses[-1], anims[b].poses[0]
        assert 0 < phase_step(last, first) <= 1 / 8 + 1e-9
        assert abs(math.remainder(first.yaw - last.yaw, math.tau)) < math.radians(20)

    turn = anims["turn_right_to_left"].poses
    assert turn[0].yaw < math.pi / 4 and turn[-1].yaw > 3 * math.pi / 4
    # Turning passes the fish's face towards the viewer (yaw through pi/2).
    assert any(abs(p.yaw - math.pi / 2) < 0.3 for p in turn)


def test_showcase_loop(sprite):
    seq = sprite.showcase(swim_loops=1)
    assert len(seq) == sum(len(a.frames) for a in sprite.animations.values())


def test_parse_size():
    assert parse_size("32") == (32, 32)
    assert parse_size("48x24") == (48, 24)


def test_cli_exports_everything(tmp_path):
    assert main(["generate", "--seed", "5", "--size", "24", "--out", str(tmp_path), "--flat"]) == 0
    data = json.loads((tmp_path / "fish_24px_5.json").read_text())
    assert set(data["animations"]) == {n for n in SHOWCASE_ORDER if not n.startswith("puff")}
    assert data["meta"]["frame_size"] == [24, 24]
    tres = (tmp_path / "fish_24px_5.tres").read_text()
    assert tres.startswith('[gd_resource type="SpriteFrames"')
    assert tres.count("[sub_resource") == sum(len(a["frames"]) for a in data["animations"].values())
    assert (tmp_path / "fish_24px_5.png").exists()
    assert (tmp_path / "gif" / "showcase.gif").exists()
    assert len(list((tmp_path / "frames").rglob("*.png"))) == tres.count("[sub_resource")


def test_species_names_are_forgiving():
    import argparse

    from fishgen.cli import parse_species

    assert parse_species("neon-tetra") == parse_species("NEON_TETRA") == "Neon tetra"
    assert parse_species("great white") == "Great white shark"
    with pytest.raises(argparse.ArgumentTypeError, match="Yellow tang, Blue tang"):
        parse_species("tang")


def test_smoothness_multiplies_frames_and_fps_at_the_same_speed():
    from fishgen.species import make_genome

    assert smooth_timing(8, 10, 10.0, 3) == (24, 30, 30.0)
    assert smooth_timing(8, 10, 10.0, 1) == (8, 10, 10.0)
    swim, _, fps = smooth_timing(8, 10, 10.0, 1.3)  # rounded frames: the fps follows, so a beat lasts as long
    assert swim / fps == pytest.approx(8 / 10.0)
    s = generate(1, 24, 24, genome=make_genome("Clownfish", 1), swim_frames=2, turn_frames=2, smoothness=2)
    assert len(s.animations["swim_right"].frames) == 4 and len(s.animations["turn_right_to_left"].frames) == 4
    assert s.animations["swim_right"].fps == 20.0


def test_sprite_command_makes_a_square_sprite(tmp_path):
    """``fishgen sprite NAME SIZE`` is the quick way to one N x N sprite of a species."""
    assert main(["sprite", "neon-tetra", "24", "--seed", "3", "--swim-frames", "2", "--turn-frames", "2",
                 "--out", str(tmp_path)]) == 0
    out = tmp_path / "neon_tetra_24px_3"
    data = json.loads((out / "neon_tetra_24px_3.json").read_text())
    assert data["meta"]["frame_size"] == [24, 24]
    assert (out / "gif" / "showcase.gif").exists() and not (out / "frames").exists()
    # Without a size it is 64 x 64, and every animal gets the same canvas, not drawn to scale.
    assert main(["sprite", "Humpback whale", "--seed", "1", "--swim-frames", "2", "--turn-frames", "2",
                 "--formats", "sheet", "--out", str(tmp_path)]) == 0
    whale = json.loads((tmp_path / "humpback_whale_64px_1" / "humpback_whale_64px_1.json").read_text())
    assert whale["meta"]["frame_size"] == [64, 64]


def test_neon_tetra_tail_reads_in_pixel_art():
    """The real tail is nearly clear; the sprite still needs a visible forked tail at 32 px."""
    from fishgen.species import make_genome

    s = generate(1, 32, 32, genome=make_genome("Neon tetra", 1), swim_frames=4, turn_frames=2)
    for f in s.animations["swim_right"].frames:
        a = f[..., 3] > 0
        back = np.where(a.any(0))[0][0]
        assert a[:, back:back + 3].any(1).sum() >= 6


def test_species_presets_render():
    from fishgen.species import SPECIES_NAMES, make_genome

    for name in SPECIES_NAMES:
        s = generate(1, 24, 24, genome=make_genome(name, 1), scale_by_size=False, swim_frames=2, turn_frames=2)
        assert opaque(s.animations["swim_right"].frames[0]).sum() > 20


def test_forced_archetype_and_dict_roundtrip():
    g = FishGenome.random(9, archetype="deep")
    assert g.height >= 0.5
    g.colors.accent_color = "#ff0000"
    assert FishGenome.from_dict(json.loads(json.dumps(g.to_dict()))) == g


def test_colour_overrides_reach_palette():  # the GUI's swatches and the JSON's key colours
    from fishgen.palette import ACCENT, FIN, build_palette

    g = FishGenome.random(4)
    g.colors.fin_color, g.colors.accent_color = "#102030", "#a0b0c0"
    pal = build_palette(g.colors)
    assert pal.colors[FIN] == (0x10, 0x20, 0x30) and pal.colors[ACCENT] == (0xa0, 0xb0, 0xc0)


def test_every_species_fits_canvas():
    from fishgen.species import SPECIES_NAMES, make_genome

    for name in SPECIES_NAMES:
        s = generate(2, 32, 32, genome=make_genome(name, 2), scale_by_size=False, swim_frames=4, turn_frames=4)
        for anim in s.animations.values():
            for f in anim.frames:
                a = f[..., 3]
                assert (a > 0).sum() > 12, name
                # Nothing is cut off at the canvas edge.
                border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
                assert (border == 0).all(), name


def test_old_genome_dicts_still_load():
    d = FishGenome.random(3).to_dict()
    for key in ("profile", "beak", "wings", "arms", "spines", "dorsal2", "hammer", "turn_lag", "silver", "scales"):
        d.pop(key)
    d["colors"].pop("fin_tips")
    d["size"] = 1.2  # a gene that no longer exists
    g = FishGenome.from_dict(d)
    assert g.kind == "fish" and g.height == FishGenome.random(3).height


def test_whales_beat_up_and_down():
    from fishgen.geometry import FishModel, Pose
    from fishgen.species import make_genome

    orca = FishModel(make_genome("Orca", 1))
    _, pos, heading = orca.spine(Pose(phase=0.25))
    assert np.ptp(pos[:, 1]) > 0.02 and np.ptp(heading) < 1e-9
    fish = FishModel(FishGenome.random(1))
    _, pos, heading = fish.spine(Pose(phase=0.25))
    assert np.ptp(pos[:, 1]) == 0 and np.ptp(heading) > 0.02


def test_new_body_parts_are_built():
    from fishgen.geometry import ARM, WING_NEAR, FishModel
    from fishgen.species import make_genome

    assert (FishModel(make_genome("Octopus", 1)).mesh.tri_part == ARM).any()
    assert (FishModel(make_genome("Cownose ray", 1)).mesh.tri_part == WING_NEAR).any()
    assert (FishModel(make_genome("Hammerhead shark", 1)).mesh.tri_part == WING_NEAR).any()
    urchin = FishModel(make_genome("Purple sea urchin", 1))
    assert len(urchin._spine_lines(0.0)) == urchin.g.spines.count > 0 and not urchin.eyes


# What every animal of a kind must have, whatever the seed.
KIND_SIGNATURE = {
    "fish": lambda g: g.tail_shape in ("forked", "rounded", "truncate", "lunate", "pointed", "fan", "none")
    and not g.wings.present and not g.arms.count and not g.spines.count and not g.swim_vertical,
    "shark": lambda g: g.tail_shape in ("heterocercal", "lunate") and g.gill_slits == 5 and g.dorsal.present
    and g.dorsal2.present and g.eye_style == "dark" and not g.wings.present,
    "whale": lambda g: g.tail_shape == "flukes" and g.swim_vertical and not g.anal.present and g.gill_slits == 0,
    # Rays are seen from the side like everything else: only a gentle extra tilt, never top-down.
    "ray": lambda g: g.wings.present and g.tail_shape == "none" and 0 < g.camera_tilt < 0.2 and not g.arms.count,
    "squid": lambda g: g.arms.count == 8 and g.wings.present and g.tail_shape == "none" and g.pulse > 0,
    "octopus": lambda g: g.arms.count == 8 and g.arms.tentacles == 0 and g.tail_shape == "none",
    "urchin": lambda g: g.profile == "sphere" and g.spines.count > 0 and g.eye_style == "none",
}


@pytest.mark.parametrize("kind", list(KIND_SIGNATURE))
def test_random_animals_stay_true_to_their_kind(kind):
    from fishgen.kinds import random_genome

    genomes = [random_genome(kind, seed) for seed in range(60)]
    for g in genomes:
        assert g.kind == kind and KIND_SIGNATURE[kind](g), (kind, g.seed)
    # ...while still varying from seed to seed.
    assert len({round(g.colors.hue, 3) for g in genomes}) > 10


def test_presets_belong_to_their_kind():
    from fishgen.species import RANDOM, SPECIES_NAMES, kind_of, make_genome, species_for

    for name in SPECIES_NAMES:
        g = make_genome(name, 5)
        assert g.kind == kind_of(name) and KIND_SIGNATURE[g.kind](g), name
        assert name in species_for(g.kind)
    assert make_genome(RANDOM, 5, kind="ray").kind == "ray"


def test_cli_kind_option(tmp_path):
    assert main(["generate", "--kind", "octopus", "--seed", "3", "--size", "24", "--swim-frames", "2",
                 "--turn-frames", "2", "--out", str(tmp_path),
                 "--formats", "sheet"]) == 0
    data = json.loads((tmp_path / "octopus_24px_3" / "octopus_24px_3.json").read_text())
    assert data["genome"]["kind"] == "octopus"
    with pytest.raises(SystemExit):
        main(["generate", "--kind", "shark", "--species", "Orca", "--out", str(tmp_path)])


def test_every_species_varies_with_the_seed():
    """Two animals of one species are never the same: shape and colour both change."""
    from fishgen.species import SPECIES_NAMES, make_genome

    for name in SPECIES_NAMES:
        gs = [make_genome(name, seed) for seed in range(8)]
        assert len({g.height for g in gs}) == 8, name
        assert len({(round(g.colors.hue, 4), round(g.colors.value, 4)) for g in gs}) == 8, name
        assert len({g.kind for g in gs}) == 1, name


def test_animals_are_drawn_to_scale():
    """At real scale an animal's drawn length follows its real length, ``px_per_m`` to the metre."""
    from fishgen.animation import px_per_unit
    from fishgen.geometry import FishModel
    from fishgen.species import make_genome

    lengths = {}
    for name in ("Koi", "Great white shark", "Bottlenose dolphin", "Tuna"):
        g = make_genome(name, 1)
        drawn = px_per_unit(g, 50.0, max_px=1600) * FishModel(g, detail=24).real_extent()
        assert drawn == pytest.approx(g.length_cm / 100 * 50.0)
        lengths[name] = (g.length_cm, drawn)
    order = sorted(lengths, key=lambda n: lengths[n][0])
    assert order == sorted(lengths, key=lambda n: lengths[n][1])
    # By default a 1 m animal is 16 px long, from 16 px for the smallest to 128 px for whales.
    assert px_per_unit(g) * FishModel(g, detail=24).real_extent() == pytest.approx(
        min(128, max(16, g.length_cm / 100 * 16)))
    # Whales are shrunk to fit the biggest sprite, 128 px.
    whale = generate(1, genome=make_genome("Humpback whale", 1), scale_by_size=True, swim_frames=2, turn_frames=2)
    assert max(whale.size) <= 128 and max(whale.size) >= 120
    # Tiny animals stop at the minimum size; the canvas fits the animal, not a fixed square.
    tetra = make_genome("Neon tetra", 1)
    assert px_per_unit(tetra, 100.0, min_px=16) * FishModel(tetra, detail=24).real_extent() == pytest.approx(16)
    s = generate(1, genome=make_genome("Koi", 1), scale_by_size=True, px_per_m=40.0, swim_frames=2, turn_frames=2)
    w, h = s.size
    assert w > h and 20 < w < 60
    assert generate(1, 32, 32, genome=make_genome("Great white shark", 1), scale_by_size=False,
                    swim_frames=2, turn_frames=2).size == (32, 32)


def test_real_lengths_are_realistic():
    from fishgen.species import make_genome

    def cm(name):
        return [make_genome(name, s).length_cm for s in range(6)]

    assert max(cm("Neon tetra")) < 5 < min(cm("Clownfish")) and max(cm("Clownfish")) < 15
    assert 300 < min(cm("Great white shark")) and max(cm("Humpback whale")) < 1800 < min(cm("Humpback whale")) * 2
    assert min(cm("Whale shark")) > max(cm("Great white shark"))
    assert len({round(x, 3) for x in cm("Koi")}) == 6  # individuals differ in size


def test_species_come_in_colour_varieties():
    from fishgen.species import make_genome, morphs_of

    assert {"Kohaku", "Showa", "Ogon"} <= set(morphs_of("Koi"))
    seen = {make_genome("Koi", s).morph for s in range(40)}
    assert len(seen) >= 4
    showa = make_genome("Koi", 3, morph="Showa")
    assert showa.morph == "Showa" and showa.colors.value < 0.3
    with pytest.raises(ValueError):
        make_genome("Koi", 3, morph="Tabby")
    assert make_genome("Tiger shark", 1).morph == ""


def test_rays_show_their_pale_underside_on_the_upstroke():
    """Seen from the side, a raised near wing turns its white belly to the viewer."""
    from fishgen.geometry import FishModel, Pose, WING_NEAR
    from fishgen.species import make_genome

    g = make_genome("Manta ray", 1, morph="Chevron")
    m = FishModel(g, detail=48)
    near = np.unique(m.mesh.tris[m.mesh.tri_part == WING_NEAR])
    pitch = 0.25 + g.camera_tilt
    under = []
    for ph in np.linspace(0, 1, 8, endpoint=False):
        cam, nrm, *_ = m.posed(Pose(phase=ph, pitch=pitch))
        under.append((nrm[near][:, 2] < 0).mean())  # wing normals point out of the top
    # On the downstroke we see the top of the near wing; as it rises, much of it turns its underside to us.
    assert max(under) > 0.4 and min(under) < 0.1
    # The wing beats through a big arc: its lowest point on screen swings by well over a quarter body length.
    ys = [m.posed(Pose(phase=ph, pitch=pitch))[0][near, 1].min() for ph in np.linspace(0, 1, 8, endpoint=False)]
    assert np.ptp(ys) > 0.3


def test_morph_option_on_the_command_line(tmp_path):
    assert main(["generate", "--species", "Koi", "--morph", "Ogon", "--seed", "2", "--size", "24",
                 "--swim-frames", "2", "--turn-frames", "2", "--formats", "sheet", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "koi_24px_2" / "koi_24px_2.json").read_text())
    assert data["genome"]["morph"] == "Ogon"
    with pytest.raises(SystemExit):
        main(["generate", "--species", "Koi", "--morph", "Tabby", "--out", str(tmp_path)])


def test_pufferfish_puffs_up_with_spikes():
    from fishgen.geometry import FishModel, Pose
    from fishgen.species import make_genome

    g = make_genome("Pufferfish", 1)
    s = generate(1, 48, 48, genome=g, scale_by_size=False)
    assert {"puff_right", "puff_left"} <= set(s.animations)
    puff = s.animations["puff_right"]
    assert puff.next == "swim_right" and not puff.loop
    assert puff.poses[0].inflate == 0 and max(p.inflate for p in puff.poses) == 1.0
    # Puffed up, it is far bigger than when calm, and only then are the spikes out.
    calm = opaque(puff.frames[0]).sum()
    big = max(opaque(f).sum() for f in puff.frames)
    assert big > 2 * calm
    # The face stays round when puffed, not a flat wall across the front.
    a = opaque(max(puff.frames, key=lambda f: opaque(f).sum()))
    cols, rows = np.where(a.any(0))[0], np.where(a.any(1))[0]
    assert a[:, cols[-3]:].any(1).sum() < 0.45 * len(rows)
    model = FishModel(g)
    assert len(model._spine_lines(0.0, 0.0)) == 0 and len(model._spine_lines(0.0, 1.0)) == g.puff_spines
    # Other fish don't puff.
    assert "puff_right" not in generate(1, 24, 24, genome=make_genome("Koi", 1), scale_by_size=False,
                                        swim_frames=2, turn_frames=2).animations


def test_orcas_come_as_males_and_females():
    from fishgen.species import make_genome

    fins = {(g.dorsal.shape, g.dorsal.height > 0.25) for g in (make_genome("Orca", s) for s in range(12))}
    assert fins == {("triangle", True), ("falcate", False)}


def test_great_white_has_teeth():
    from fishgen.species import make_genome

    g = make_genome("Great white shark", 1)
    assert g.teeth and g.mouth > 0
    s = generate(1, 400, 200, genome=g, scale_by_size=False, swim_frames=2, turn_frames=2, style="smooth")
    f = s.animations["swim_right"].frames[0]
    # The open jaw (near-black gape) with pale teeth in it shows on the side of the head, at the front.
    head = f[:, f.shape[1] // 2:]
    rgb = head[..., :3].astype(int)
    on = head[..., 3] == 255
    gape = on & (rgb.max(axis=-1) < 50)
    teeth = on & (rgb.min(axis=-1) > 170)
    assert gape.sum() >= 3 and teeth.sum() >= 2

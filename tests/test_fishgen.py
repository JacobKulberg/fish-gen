import json
import math

import numpy as np
import pytest

from fishgen import FishGenome, generate
from fishgen.animation import SHOWCASE_ORDER, build_animations
from fishgen.cli import main, parse_size
from fishgen.palette import OUTLINE, TRANSPARENT


@pytest.fixture(scope="module")
def sprite():
    return generate(42, 32, 32)


def test_same_seed_same_fish():
    a, b = generate(7, 24, 24), generate(7, 24, 24)
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
    s = generate(3, 48, 32)
    assert s.animations["swim_right"].frames[0].shape == (32, 48, 4)


def test_fish_fits_inside_canvas_with_outline(sprite):
    # Every frame is drawn, and nothing touches the canvas edge (room for the outline).
    for anim in sprite.animations.values():
        for idx in anim.indexed:
            assert (idx != TRANSPARENT).sum() > 20
            assert not idx[0].any() and not idx[-1].any()
            assert not idx[:, 0].any() and not idx[:, -1].any()


def test_silhouette_is_outlined(sprite):
    # Every opaque pixel that touches transparency must be an outline pixel.
    idx = sprite.animations["swim_right"].indexed[0]
    filled = idx != TRANSPARENT
    padded = np.pad(filled, 1)
    touches_empty = ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    assert np.all(idx[filled & touches_empty] == OUTLINE)


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
    assert len(seq) == sum(len(sprite.animations[n].frames) for n in SHOWCASE_ORDER)


def test_parse_size():
    assert parse_size("32") == (32, 32)
    assert parse_size("48x24") == (48, 24)


def test_cli_exports_everything(tmp_path):
    assert main(["generate", "--seed", "5", "--size", "24", "--out", str(tmp_path), "--flat"]) == 0
    data = json.loads((tmp_path / "fish_5.json").read_text())
    assert set(data["animations"]) == set(SHOWCASE_ORDER)
    assert data["meta"]["frame_size"] == [24, 24]
    tres = (tmp_path / "fish_5.tres").read_text()
    assert tres.startswith('[gd_resource type="SpriteFrames"')
    assert tres.count("[sub_resource") == sum(len(a["frames"]) for a in data["animations"].values())
    assert (tmp_path / "fish_5.png").exists()
    assert (tmp_path / "gif" / "showcase.gif").exists()
    assert len(list((tmp_path / "frames").rglob("*.png"))) == tres.count("[sub_resource")


def test_species_presets_render():
    from fishgen.species import SPECIES_NAMES, make_genome

    for name in SPECIES_NAMES:
        s = generate(1, 24, 24, genome=make_genome(name, 1))
        assert (s.animations["swim_right"].indexed[0] != TRANSPARENT).sum() > 20


def test_forced_archetype_and_dict_roundtrip():
    g = FishGenome.random(9, archetype="deep")
    assert g.height >= 0.5
    g.colors.accent_color = "#ff0000"
    assert FishGenome.from_dict(json.loads(json.dumps(g.to_dict()))) == g


def test_colour_overrides_reach_palette():
    from fishgen.palette import ACCENT, FIN, build_palette

    g = FishGenome.random(4)
    g.colors.fin_color, g.colors.accent_color = "#102030", "#a0b0c0"
    pal = build_palette(g.colors)
    assert pal.colors[FIN] == (0x10, 0x20, 0x30) and pal.colors[ACCENT] == (0xa0, 0xb0, 0xc0)


def test_every_species_fits_canvas():
    from fishgen.species import SPECIES_NAMES, make_genome

    for name in SPECIES_NAMES:
        s = generate(2, 32, 32, genome=make_genome(name, 2))
        for anim in s.animations.values():
            for idx in anim.indexed:
                assert (idx != TRANSPARENT).sum() > 20, name
                # Only the outline may reach the canvas edge: nothing is cut off.
                border = np.concatenate([idx[0], idx[-1], idx[:, 0], idx[:, -1]])
                assert np.isin(border, (TRANSPARENT, OUTLINE)).all(), name


def test_old_genome_dicts_still_load():
    d = FishGenome.random(3).to_dict()
    for key in ("profile", "beak", "wings", "arms", "spines", "dorsal2", "hammer", "turn_lag"):
        d.pop(key)
    d["colors"].pop("fin_tips")
    assert FishGenome.from_dict(d) == FishGenome.random(3)


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
    "fish": lambda g: g.tail_shape in ("forked", "rounded", "truncate", "lunate", "pointed", "fan")
    and not g.wings.present and not g.arms.count and not g.spines.count and not g.swim_vertical,
    "shark": lambda g: g.tail_shape == "heterocercal" and g.gill_slits == 5 and g.dorsal.present
    and g.dorsal2.present and g.eye_style == "dark" and not g.wings.present,
    "whale": lambda g: g.tail_shape == "flukes" and g.swim_vertical and not g.anal.present and g.gill_slits == 0,
    "ray": lambda g: g.wings.present and g.tail_shape == "none" and g.camera_tilt > 0 and not g.arms.count,
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
    assert main(["generate", "--kind", "octopus", "--seed", "3", "--size", "24", "--out", str(tmp_path),
                 "--formats", "sheet"]) == 0
    data = json.loads((tmp_path / "octopus_3" / "octopus_3.json").read_text())
    assert data["genome"]["kind"] == "octopus"
    with pytest.raises(SystemExit):
        main(["generate", "--kind", "shark", "--species", "Orca", "--out", str(tmp_path)])


def test_every_species_varies_with_the_seed():
    """Two animals of one species are never the same: shape and colour both change."""
    from fishgen.species import SPECIES_NAMES, make_genome

    for name in SPECIES_NAMES:
        gs = [make_genome(name, seed) for seed in range(8)]
        assert len({round(g.height, 4) for g in gs}) == 8, name
        assert len({(round(g.colors.hue, 4), round(g.colors.value, 4)) for g in gs}) == 8, name
        assert len({g.kind for g in gs}) == 1, name

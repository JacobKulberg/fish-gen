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

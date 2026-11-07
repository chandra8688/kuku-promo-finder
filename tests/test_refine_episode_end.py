from src.refine import refine_boundaries


def test_tail_pad_never_passes_the_episode_end():
    beats = [
        {"beat_id": "b1", "kind": "DIALOGUE", "start_s": 0.0, "end_s": 10.0},
        {"beat_id": "b2", "kind": "ACTION", "start_s": 10.0, "end_s": 29.8},
    ]  # episode ends at 29.8 s
    s, e, ops = refine_boundaries(beats, 0.0, 29.8, ["b1", "b2"])
    assert e <= 29.8

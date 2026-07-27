from aoc import PredictableContrastEProcess, pure_state_density


def test_stationary_pure_stream_does_not_create_evidence():
    reference = pure_state_density([1.0, 0.0])
    detector = PredictableContrastEProcess(
        reference,
        adaptation_window=8,
        alpha=0.01,
    )
    records = [detector.update([1.0, 0.0]) for _ in range(80)]
    assert max(record.e_value for record in records) <= 1.0 + 1e-12
    assert not records[-1].alarm
    assert abs(records[-1].anytime_p_value - 1.0) < 1e-14


def test_orthogonal_state_change_is_detected_after_witness_adapts():
    reference = pure_state_density([1.0, 0.0])
    detector = PredictableContrastEProcess(
        reference,
        adaptation_window=12,
        alpha=0.01,
    )
    for _ in range(24):
        detector.update([1.0, 0.0])
    post = [detector.update([0.0, 1.0]) for _ in range(40)]
    alarm_times = [record.time for record in post if record.alarm]
    assert alarm_times
    assert alarm_times[0] - 24 <= 12
    assert post[-1].contrast > 0.99

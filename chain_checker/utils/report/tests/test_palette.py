from chain_checker.utils.report import palette


def test_accuracy_color_is_true_color_at_and_above_eighty_percent():
    assert palette.accuracy_color(0.8) == palette.TRUE_COLOR
    assert palette.accuracy_color(1.0) == palette.TRUE_COLOR


def test_accuracy_color_is_mid_color_between_fifty_and_eighty_percent():
    assert palette.accuracy_color(0.5) == palette.MID_COLOR
    assert palette.accuracy_color(0.79) == palette.MID_COLOR


def test_accuracy_color_is_false_color_below_fifty_percent():
    assert palette.accuracy_color(0.49) == palette.FALSE_COLOR
    assert palette.accuracy_color(0.0) == palette.FALSE_COLOR

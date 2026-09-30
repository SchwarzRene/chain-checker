from chain_checker.utils.report.hover_script import HOVER_SCRIPT


def test_hover_script_is_a_real_script_tag():
    assert HOVER_SCRIPT.strip().startswith("<script>")
    assert HOVER_SCRIPT.strip().endswith("</script>")

from litalign.maths import sum


def test_sum():
    """Test the my_sum function."""
    assert sum.my_sum(2, 3) == 5
    assert sum.my_sum(-1, 1) == 0
    assert sum.my_sum(0, 0) == 0
    assert sum.my_sum(-5, -5) == -10
    assert sum.my_sum(100, 200) == 300

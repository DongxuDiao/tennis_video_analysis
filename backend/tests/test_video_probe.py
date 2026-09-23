from types import SimpleNamespace

from app.video.probe import stream_rotation


def _stream(side_data=None, metadata=None):
    return SimpleNamespace(side_data=side_data, metadata=metadata or {})


def test_stream_rotation_accepts_pyav_decoded_float_angle():
    assert stream_rotation(_stream({"DISPLAYMATRIX": -180.0})) == 180
    assert stream_rotation(_stream({"DISPLAYMATRIX": -90.0})) == 270
    assert stream_rotation(_stream({"DISPLAYMATRIX": 0.0})) == 0


def test_stream_rotation_falls_back_when_side_data_is_not_bytes():
    assert stream_rotation(
        _stream({"DISPLAYMATRIX": object()}, {"rotate": "90"})
    ) == 90

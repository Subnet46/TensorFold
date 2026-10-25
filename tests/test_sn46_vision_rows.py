"""(patch sn46/vision-rows) Row statistics on image prompts: token-id prompts carry their pictures, the measured spec
refuses prompt state for them, and the encode is fingerprinted where two hosts could part."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.server.errors import RequestError
from tensorfold.sn46 import rows
from tensorfold.vision.glm import GlmVision, GlmVisionLimits
from tensorfold.vision.images import ImageInputError

IMAGE = 154854


def vision() -> GlmVision:
    """The frontend without its tower: prepare_ids only fits pictures and counts their rows."""
    front = object.__new__(GlmVision)
    front.image_token = IMAGE
    front.limits = GlmVisionLimits()
    return front


def picture(h: int, w: int, fill: int = 0):
    return torch.full((3, h, w), fill, dtype=torch.uint8)


def test_prepare_ids_takes_expanded_ids_whose_runs_match_the_pictures():
    front = vision()
    a, b = picture(56, 84), picture(28, 56, 7)            # 2 x 3 = 6 rows, 1 x 2 = 2 rows
    ids = [1, 8, *[IMAGE] * 6, 9, 2, 8, *[IMAGE] * 2, 9, 3]
    prepared = front.prepare_ids(ids, [a, b])
    assert prepared.token_ids == ids
    assert prepared.positions == [2, 3, 4, 5, 6, 7, 11, 12]
    assert [kind for kind, _ in prepared.items] == ["image", "image"]
    assert prepared.content_hash == GlmVision._hash(prepared.items)
    assert front.prepare_ids(ids, [a, picture(28, 56, 8)]).content_hash != prepared.content_hash


@pytest.mark.parametrize("ids", [
    [1, 8, IMAGE, 9, 2],                                   # compact: one id a picture
    [1, 8, *[IMAGE] * 2, 9, 2, 8, *[IMAGE] * 6, 9],        # the runs in the other order
    [1, 8, *[IMAGE] * 8, 9],                               # one run for two pictures
    [1, 2, 3],                                             # no image ids at all
])
def test_prepare_ids_refuses_ids_that_do_not_carry_the_pictures_rows(ids):
    with pytest.raises(ImageInputError, match="image runs"):
        vision().prepare_ids(ids, [picture(56, 84), picture(28, 56)])


def test_a_measured_image_prompt_keeps_no_prompt_state():
    rows.ACTIVE = True
    try:
        rows.arm(rows.Spec(record=True, prompt=[1, 2], cache=True))
        with pytest.raises(rows.RowsError, match="cache off"):
            rows.take_armed([1, 2], True, images=True)
        assert rows.disarm() is None
        spec = rows.Spec(record=True, prompt=[1, 2], cache=False)
        rows.arm(spec)
        assert rows.take_armed([1, 2], True, images=True) is spec
        rows.arm(rows.Spec(record=True, prompt=[1, 2]))
        with pytest.raises(rows.RowsError, match="without a grammar"):
            rows.take_armed([1, 2], False, images=True)
    finally:
        rows.disarm()
        rows.ACTIVE = False


def test_feature_hash_is_the_rows_bits():
    pytest.importorskip("triton")                           # the CUDA engine module imports it
    from tensorfold.families.glm5_next.cuda.engine import feature_hash

    features = torch.randn(4, 6).to(torch.bfloat16)
    same = features.clone()
    assert feature_hash(features) == feature_hash(same)
    flipped = same.view(torch.int16).clone()
    flipped[2, 3] ^= 1                                      # one bit of one row
    assert feature_hash(flipped.view(torch.bfloat16)) != feature_hash(features)
    assert feature_hash(features.reshape(6, 4)) != feature_hash(features)


def png_url() -> str:
    pil = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    pil.new("RGB", (64, 48), (200, 30, 30)).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


class Frontend:
    def __init__(self):
        self.loaded, self.calls = [], []

    def load_images(self, sources):
        self.loaded = [s.url for s in sources]
        return ["fitted"] * len(sources)

    def prepare_ids(self, ids, images):
        self.calls.append((list(ids), images))
        return SimpleNamespace(token_ids=list(ids), positions=[1])


def test_token_id_prompt_images_reach_the_frontend():
    front = Frontend()
    app = SimpleNamespace(vision=front)
    prepared = App._id_images(app, [1, IMAGE, 2], [png_url()])
    assert prepared.positions == [1]
    assert front.calls == [([1, IMAGE, 2], ["fitted"])]


@pytest.mark.parametrize(("frontend", "images", "message"), [
    (None, ["data:image/png;base64,AA=="], "--vision"),
    (Frontend(), [], "non-empty list"),
    (Frontend(), "data:image/png;base64,AA==", "non-empty list"),
    (Frontend(), [3], "non-empty list"),
])
def test_token_id_prompt_images_are_refused_with_a_reason(frontend, images, message):
    with pytest.raises(RequestError, match=message):
        App._id_images(SimpleNamespace(vision=frontend), [1, 2], images)


def test_a_frontend_refusal_is_a_request_error():
    class Refusing(Frontend):
        def prepare_ids(self, ids, images):
            raise ImageInputError("the prompt's image runs [1] do not match its pictures' rows [6]")

    with pytest.raises(RequestError, match="image runs"):
        App._id_images(SimpleNamespace(vision=Refusing()), [1, IMAGE], [png_url()])


def test_prepared_request_carries_vision():
    assert "vision" in PreparedRequest.__dataclass_fields__


class MarkerTokenizer:
    """The image markers as their ids, every other character as its code point."""

    MARKS: ClassVar[dict[str, int]] = {"<|begin_of_image|>": 8, "<|image|>": IMAGE, "<|end_of_image|>": 9}

    def encode(self, text, add_special_tokens=False):
        out = []
        while text:
            mark = next((m for m in self.MARKS if text.startswith(m)), None)
            out.append(self.MARKS[mark] if mark else ord(text[0]))
            text = text[len(mark) if mark else 1:]
        return SimpleNamespace(ids=out)


def test_ids_from_prepare_round_trip_through_prepare_ids():
    """What ``/tokenize`` returns for chat messages (``prepare``'s ids) is what a token-id request carries."""
    front = vision()
    front.tok = MarkerTokenizer()
    pictures = [picture(56, 84, 1), picture(84, 28, 2)]
    span = "<|begin_of_image|><|image|><|end_of_image|>"
    chat = front.prepare(f"look {span} and {span}?", pictures)
    again = front.prepare_ids(chat.token_ids, pictures)
    assert (again.token_ids, again.positions, again.content_hash) == \
        (chat.token_ids, chat.positions, chat.content_hash)


def prepared_app(tmp_path, frontend):
    """An App with just what ``_prepare`` reads (the pattern of test_cuda_response_format)."""
    import json as _json
    import threading

    from tensorfold.cuda import server

    (tmp_path / "tokenizer_config.json").write_text(_json.dumps(
        {"chat_template": "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:"}))
    app = server.App.__new__(server.App)
    app.engine = SimpleNamespace()
    app.vision = frontend
    app.served = "fake-cuda"
    app.model_dir = tmp_path
    app.tok = SimpleNamespace()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking = False
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 64
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def test_a_token_id_completion_with_images_passes_the_modality_check(tmp_path):
    front = Frontend()
    app = prepared_app(tmp_path, front)
    prepared = app._prepare({"prompt": [1, IMAGE, 2], "images": [png_url()], "max_tokens": 4}, False)
    assert prepared.prompt == [1, IMAGE, 2] and prepared.vision.positions == [1]
    assert front.calls == [([1, IMAGE, 2], ["fitted"])]
    with pytest.raises(RequestError, match="text only"):           # anywhere else, images stay refused
        app._prepare({"messages": [{"role": "user", "content": "hi"}], "images": [png_url()]}, True)
    with pytest.raises(RequestError, match="text only"):
        app._prepare({"prompt": "text", "images": [png_url()]}, False)


def test_a_template_that_writes_glm_markers_is_used_as_it_is(tmp_path):
    """The SN46 checkpoints' template emits GLM's image markers itself: --vision keeps it; one with neither the
    markers nor the stock reminder is still refused."""
    from tensorfold.families.glm5_next.cuda.app import media_template

    inner = SimpleNamespace(template="unchanged")
    (tmp_path / "chat_template.jinja").write_text(
        "{%- macro emit_image() -%}<|begin_of_image|><|image|><|end_of_image|>{%- endmacro -%}{{ messages }}")
    media_template(inner, tmp_path)
    assert inner.template == "unchanged"
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    with pytest.raises(ValueError, match="no media branch"):
        media_template(inner, tmp_path)


def test_image_rows_are_the_rows_prepare_ids_expects(tmp_path):
    """The count the SN46 contract asks for is the run length ``prepare_ids`` then checks."""
    pil = pytest.importorskip("PIL.Image")
    pytest.importorskip("torchvision")                       # fitting resizes with it
    front = vision()
    front.allow_urls = False

    def url(w, h):
        out = io.BytesIO()
        pil.new("RGB", (w, h), (10, 200, 30)).save(out, format="PNG")
        return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()

    from tensorfold.vision.images import ImageSource

    sources = [ImageSource(url(640, 480)), ImageSource(url(64, 64))]
    rows = front.image_rows(sources)
    assert len(rows) == 2 and all(n > 0 for n in rows)
    ids = [1, 8, *[IMAGE] * rows[0], 9, 2, 8, *[IMAGE] * rows[1], 9]
    prepared = front.prepare_ids(ids, front.load_images(sources))
    assert len(prepared.positions) == sum(rows)

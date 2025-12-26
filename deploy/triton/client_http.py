from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import Request, urlopen

import fire
import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


@dataclass(frozen=True)
class InferenceResult:
    pred: int
    scores: List[float]


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def infer(
    server_url: str = "http://localhost:8000",
    model_name: str = "fashion_mnist",
    test_index: int = 0,
) -> InferenceResult:
    ds = FashionMNIST(
        root=str(Path(".cache/fashion_mnist")),
        train=False,
        download=True,
        transform=ToTensor(),
    )
    image, _ = ds[test_index]

    x = image.numpy().astype(np.float32)[None, ...]  # [1,1,28,28]

    cfg_path = Path("deploy/triton/model_repository") / model_name / "config.pbtxt"
    cfg_txt = cfg_path.read_text(encoding="utf-8")

    def _extract_name(kind: str) -> str:
        start = cfg_txt.find(f"{kind} [")
        if start < 0:
            raise RuntimeError(f"Cannot find '{kind} [' in {cfg_path}")
        name_pos = cfg_txt.find('name: "', start)
        if name_pos < 0:
            raise RuntimeError(f"Cannot find {kind} name in {cfg_path}")
        name_pos += len('name: "')
        end = cfg_txt.find('"', name_pos)
        return cfg_txt[name_pos:end]

    in_name = _extract_name("input")
    out_name = _extract_name("output")

    infer_url = f"{server_url}/v2/models/{model_name}/infer"
    payload = {
        "inputs": [
            {
                "name": in_name,
                "shape": list(x.shape),
                "datatype": "FP32",
                "data": x.reshape(-1).tolist(),
            }
        ],
        "outputs": [{"name": out_name}],
    }

    resp = _post_json(infer_url, payload)
    outputs = resp.get("outputs", [])
    if not outputs:
        raise RuntimeError(f"No outputs in response: {resp}")
    scores = outputs[0].get("data", [])
    if not scores:
        raise RuntimeError(f"No output data in response: {resp}")

    scores_f = [float(s) for s in scores]
    pred = int(np.argmax(np.array(scores_f)))

    print(f"pred={pred}")
    return InferenceResult(pred=pred, scores=scores_f)


def main() -> None:
    fire.Fire({"infer": infer})


if __name__ == "__main__":
    main()

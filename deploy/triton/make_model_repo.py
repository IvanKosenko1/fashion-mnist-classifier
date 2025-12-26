from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fire

try:
    import onnx
    from onnx import TensorProto
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Package 'onnx' is required for Triton repo generation. Install it into your venv."
    ) from exc


@dataclass(frozen=True)
class IoSpec:
    name: str
    dtype: str
    dims: List[int]


_TENSOR_DTYPE_MAP = {
    TensorProto.FLOAT: "FP32",
    TensorProto.FLOAT16: "FP16",
    TensorProto.INT64: "INT64",
    TensorProto.INT32: "INT32",
    TensorProto.UINT8: "UINT8",
}


def _dim_value(dim) -> Optional[int]:
    if hasattr(dim, "dim_value") and dim.dim_value:
        return int(dim.dim_value)
    return None


def _parse_io(value_info) -> IoSpec:
    tt = value_info.type.tensor_type
    elem_type = int(tt.elem_type)
    dtype = _TENSOR_DTYPE_MAP.get(elem_type, "FP32")

    dims: List[int] = []
    for d in tt.shape.dim:
        v = _dim_value(d)
        dims.append(v if v is not None else -1)  # -1 => dynamic
    return IoSpec(name=value_info.name, dtype=dtype, dims=dims)


def _pick_first_onnx(onnx_dir: Path) -> Path:
    candidates = sorted([p for p in onnx_dir.glob("*.onnx") if p.is_file()])
    if not candidates:
        raise FileNotFoundError(f"No .onnx files found in: {onnx_dir}")
    return candidates[0]


def _format_dims_for_triton(dims: List[int], max_batch_size: int) -> List[int]:
    if max_batch_size > 0 and len(dims) >= 1:
        return dims[1:]
    return dims


def _dims_pbtxt(dims: List[int]) -> str:
    return ", ".join(str(int(d)) for d in dims)


def _write_config_pbtxt(
    model_dir: Path,
    model_name: str,
    input_spec: IoSpec,
    output_spec: IoSpec,
    max_batch_size: int,
) -> None:
    config = f'''
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "{input_spec.name}"
    data_type: {input_spec.dtype}
    dims: [ {_dims_pbtxt(_format_dims_for_triton(input_spec.dims, max_batch_size))} ]
  }}
]

output [
  {{
    name: "{output_spec.name}"
    data_type: {output_spec.dtype}
    dims: [ {_dims_pbtxt(_format_dims_for_triton(output_spec.dims, max_batch_size))} ]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
'''.strip() + "\n"
    (model_dir / "config.pbtxt").write_text(config, encoding="utf-8")


def build(
    model_name: str = "fashion_mnist",
    onnx_path: Optional[str] = None,
    repo_root: str = "deploy/triton/model_repository",
    max_batch_size: int = 64,
) -> str:
    repo_root_path = Path(repo_root).resolve()
    repo_root_path.mkdir(parents=True, exist_ok=True)

    if onnx_path is None:
        onnx_path_obj = _pick_first_onnx(Path("artifacts/onnx").resolve())
    else:
        onnx_path_obj = Path(onnx_path).resolve()

    model_repo_dir = repo_root_path / model_name
    version_dir = model_repo_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path_obj))
    graph = model.graph

    if not graph.input or not graph.output:
        raise RuntimeError("ONNX graph has no inputs/outputs. Cannot generate Triton config.")

    input_spec = _parse_io(graph.input[0])
    output_spec = _parse_io(graph.output[0])

    (version_dir / "model.onnx").write_bytes(onnx_path_obj.read_bytes())

    _write_config_pbtxt(
        model_dir=model_repo_dir,
        model_name=model_name,
        input_spec=input_spec,
        output_spec=output_spec,
        max_batch_size=max_batch_size,
    )

    return str(model_repo_dir)


def main() -> None:
    fire.Fire({"build": build})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Weight-only fp16: store big fp32 initializers as fp16, upcast via Cast at
runtime. Compute graph stays fp32 -- no dtype reconciliation, numerics are pure
weight rounding (~1e-3 rel)."""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
casts, converted, saved = [], 0, 0
keep = list(m.graph.initializer)
del m.graph.initializer[:]
for init in keep:
    arr = numpy_helper.to_array(init)
    if init.data_type == onnx.TensorProto.FLOAT and arr.size > 100_000:
        h = numpy_helper.from_array(arr.astype(np.float16), init.name + "_h")
        m.graph.initializer.append(h)
        casts.append(helper.make_node("Cast", [init.name + "_h"], [init.name],
                                      to=onnx.TensorProto.FLOAT))
        converted += 1
        saved += arr.size * 2
    else:
        m.graph.initializer.append(init)
nodes = casts + list(m.graph.node)
del m.graph.node[:]
m.graph.node.extend(nodes)
onnx.save(m, dst)
print(f"converted {converted} tensors, saved {saved/1e6:.0f} MB")

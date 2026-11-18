"""Download Pipecat's backchannel detector and print exactly what it takes in and gives out.

    uv run python -m scripts.probe_backchannel

Run it once. The output is what we need to plug the model into the live call.
"""

import json

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "pipecat-ai/backchannel-detector"


def main() -> None:
    config = json.load(open(hf_hub_download(REPO, "model.json")))
    print("model.json:")
    print(json.dumps(config, indent=2)[:3000])
    session = ort.InferenceSession(hf_hub_download(REPO, "backchannel-int8.onnx"), providers=["CPUExecutionProvider"])
    print("\nonnx inputs: ", [(i.name, i.shape, i.type) for i in session.get_inputs()])
    print("onnx outputs:", [(o.name, o.shape, o.type) for o in session.get_outputs()])

    feed = {}
    for i in session.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        feed[i.name] = np.zeros(shape, dtype=np.float32)
    outputs = session.run(None, feed)
    print(
        "\none call on silence:",
        {
            o.name: np.asarray(v).round(3).tolist()
            for o, v in zip(session.get_outputs(), outputs, strict=True)
            if o.name in ("intent", "vad")
        },
    )


if __name__ == "__main__":
    main()

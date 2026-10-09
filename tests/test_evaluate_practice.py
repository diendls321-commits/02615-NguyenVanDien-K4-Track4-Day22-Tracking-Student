"""Kiểm tra tương thích NumPy khi gọi tiến trình chấm riêng."""

import json

from evaluate_practice import run_trackeval


def test_run_trackeval_patches_numpy_in_child(tmp_path):
    """Tiến trình con có alias NumPy và nhận nguyên vẹn tham số chấm.

    Args:
        tmp_path: Thư mục tạm do pytest cấp để tạo script chấm giả.
    """
    root = tmp_path / "thư mục chấm có khoảng trắng"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    entrypoint = scripts / "run_mot_challenge.py"
    entrypoint.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "assert np.array(['1.5'], dtype=np.float)[0] == 1.5\n"
        "assert np.array(['2'], dtype=np.int)[0] == 2\n"
        "assert __name__ == '__main__'\n"
        "Path(__file__).with_suffix('.json').write_text(json.dumps(sys.argv))\n",
        encoding="utf-8",
    )

    run_trackeval(root, "lan_cham", "LAB", "train")

    args = json.loads(entrypoint.with_suffix(".json").read_text())
    assert args[0] == str(entrypoint)
    assert args[args.index("--GT_FOLDER") + 1] == str(root / "data" / "gt" / "mot_challenge")
    assert args[args.index("--BENCHMARK") + 1] == "LAB"
    assert args[args.index("--SPLIT_TO_EVAL") + 1] == "train"
    assert args[args.index("--SEQ_INFO") + 1] == "video_1"
    assert args[args.index("--TRACKERS_TO_EVAL") + 1] == "lan_cham"
    assert args[args.index("--METRICS") + 1:args.index("--USE_PARALLEL")] == [
        "HOTA", "CLEAR", "Identity"
    ]

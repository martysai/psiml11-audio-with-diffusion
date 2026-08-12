# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test-collection-time setup, shared by every test module under tests/.
pytest imports conftest.py before collecting/importing any test module,
which both of the following rely on.

1. Puts src/ on sys.path so `import audioseal_robust` / `import audioseal`
   resolve without requiring PYTHONPATH=src to be set manually -- this repo
   has no pyproject.toml/setup.py installing the packages under src/.

2. Sets NO_TORCH_COMPILE=1 (audioseal's own escape hatch -- see
   audioseal/libs/moshi/utils/compile.py:torch_compile_lazy) before
   audioseal.models is ever imported. AudioSealWM.get_watermark is
   torch.compile-decorated at import time, and torch.compile's C++ codegen
   backend needs an MSVC cl.exe, which this Windows box doesn't have; without
   this, any test that calls get_watermark fails with
   InductorError: InvalidCxxCompiler at first call, not at import time --
   but the decorator itself reads the env var when torch_compile_lazy runs
   (at import), so this must happen before import, not inside a fixture.
   setdefault so it doesn't clobber a value the caller already set.
"""

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("NO_TORCH_COMPILE", "1")

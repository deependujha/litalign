# Based on the original setup from Lightning AI's litdata library.
# Modified by Deependu Jha for use in litalign.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import time

__version__ = "0.0.3"
__author__ = "Deependu Jha"
__author_email__ = "deependujha21@gmail.com"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2023-{time.strftime('%Y')}, {__author__}."
__homepage__ = "https://github.com/deependujha/litalign"
__docs_url__ = "https://github.com/deependujha/litalign"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = "Lightning-native library for fine-tuning LLMs using alignment techniques like PPO, DPO, GRPO and more"
__long_doc__ = """
What is it?
-----------

TBD @deependujha
"""  # TODO

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__docs_url__",
    "__homepage__",
    "__license__",
    "__version__",
]

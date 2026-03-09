import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any


PROMPT = "Can you solve the following math problem? "
BASE = " Put your final answer within \\boxed{{}}."
COT = " Please reason step by step, and put your final answer within \\boxed{{}}."
COT_ALT = " Please explain your reasoning with a detailed, step-by-step solution, and present your final answer within \\boxed{{}}."
GPQA_QUERY_TEMPLATE = "Answer the following multiple choice question. The last line of your response should be of the following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}'). Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
STEP_DELIM = "<SE>"
COT_STEP = (
    " Please reason step by step.\n\n"
    "You MUST follow this exact format:\n\n"
    "Step 1: <reasoning for step 1>\n"
    "<SE>\n"
    "Step 2: <reasoning for step 2>\n"
    "<SE>\n"
    "...\n\n"
    "Rules:\n"
    f"1. Every step MUST end with {STEP_DELIM} on its own line.\n"
    "2. Do NOT skip the delimiter.\n"
    "3. After finishing all steps, give the final answer in \\boxed{}.\n"
)

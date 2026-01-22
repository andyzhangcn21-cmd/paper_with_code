# data/linear_tasks.py
from __future__ import annotations

# 这个文件只做兼容：把旧名字/旧导入路径统一转到 synthetic_tasks.py
from data.synthetic_tasks import (
    PolynomialTask,
    generate_polynomial_tasks,
    make_grf_batch,
    make_sparse_linear_batch,
)

__all__ = [
    "PolynomialTask",
    "generate_polynomial_tasks",
    "make_grf_batch",
    "make_sparse_linear_batch",
]

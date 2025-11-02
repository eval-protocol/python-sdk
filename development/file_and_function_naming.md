# 文件名和函数名的自动发现

## 总结

使用 `@evaluation_test` 装饰器后：

### ✅ 函数名：完全自动处理
- 任何函数名都可以，不需要以 `test_` 开头
- Decorator 会自动注册正确的测试名称
- **无需任何配置或命令行参数**

### ✅ 文件名：三种方式

#### 方式 1：明确指定文件路径（推荐）
最简单直接，任何文件名都可以：

```bash
# 运行特定文件，任何文件名都可以
pytest path/to/my_evaluation.py -v
pytest examples/my_custom_file.py -v
pytest evals/math_eval.py -v
```

#### 方式 2：使用标准命名（传统方式）
文件名符合 `test_*.py` 或 `*_test.py`：

```bash
# 自动发现
pytest  # 会发现所有 test_*.py 文件
```

#### 方式 3：使用 --ep-discover-all 标志
让 pytest 搜索所有 Python 文件：

```bash
pytest --ep-discover-all -v
```

## 完整示例

### 文件: `examples/my_evaluation.py` （任意文件名）

```python
from eval_protocol.pytest import evaluation_test
from eval_protocol.models import EvaluationRow, EvaluateResult

# 函数名也可以是任意的
@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[{"role": "user", "content": "Test"}])
    ]]
)
async def my_custom_function(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = EvaluateResult(score=1.0)
    return row
```

### 运行方式

```bash
# 方式 1：明确指定文件（推荐）
pytest examples/my_evaluation.py -v

# 方式 2：使用 --ep-discover-all
pytest examples/ --ep-discover-all -v

# 方式 3：运行整个目录（如果文件名是 test_*.py）
pytest examples/  # 只会发现 test_*.py 文件
```

## 实际效果

```bash
$ pytest examples/my_evaluation.py --collect-only -v

collected 1 item

<Module my_evaluation.py>
  <Coroutine test_my_custom_function[rows(len=1)]>  # 自动注册！

$ pytest examples/my_evaluation.py -v

============================== 1 passed in 0.08s ===============================
```

## 最佳实践

### 推荐做法 👍

**选项 A：使用标准命名**
```
tests/
  test_math_evaluation.py     # ✅ 标准命名
  test_coding_evaluation.py   # ✅ 标准命名
```

运行：`pytest tests/`

**选项 B：任意命名 + 明确指定**
```
evals/
  math.py           # ✅ 简洁命名
  coding.py         # ✅ 简洁命名
  reasoning.py      # ✅ 简洁命名
```

运行：`pytest evals/math.py evals/coding.py evals/reasoning.py`

或创建一个脚本：
```bash
#!/bin/bash
# run_evals.sh
pytest evals/math.py evals/coding.py evals/reasoning.py "$@"
```

### 函数命名建议

虽然函数名可以是任意的，但建议使用描述性名称：

```python
# ✅ 好的命名 - 描述性强
@evaluation_test(...)
async def evaluate_math_accuracy(row: EvaluationRow) -> EvaluationRow:
    ...

# ✅ 也可以 - 使用传统 test_ 前缀
@evaluation_test(...)
async def test_math_accuracy(row: EvaluationRow) -> EvaluationRow:
    ...

# ⚠️ 可以但不推荐 - 不够描述性
@evaluation_test(...)
async def eval1(row: EvaluationRow) -> EvaluationRow:
    ...
```

## 配置示例

### pytest.ini

如果你想让 pytest 自动发现所有文件，可以修改配置：

```ini
[pytest]
# 发现所有 Python 文件
python_files = *.py

# 或者指定多个模式
python_files = test_*.py *_test.py eval_*.py

# 函数名模式（我们已经自动处理了，这个可以保持默认）
python_functions = test_*
```

### pyproject.toml

```toml
[tool.pytest.ini_options]
python_files = ["*.py"]
python_functions = ["test_*"]
```

## 技术细节

### 函数名自动处理机制

1. 当使用 `@evaluation_test` 装饰函数时
2. Decorator 检查函数名是否以 `test_` 开头
3. 如果不是，自动在模块的全局命名空间中注册 `test_{function_name}` 别名
4. Pytest 扫描模块时发现这个别名，识别为测试

### 文件名处理

- Pytest 通过文件名模式匹配来决定扫描哪些文件
- 默认只扫描 `test_*.py` 和 `*_test.py`
- 使用 `--ep-discover-all` 会修改这个配置为 `*.py`
- 明确指定文件路径时，不受文件名限制

## 总结

| 场景 | 函数名 | 文件名 | 命令 |
|------|--------|--------|------|
| 完全标准 | `test_*` | `test_*.py` | `pytest` |
| 任意命名 + 明确路径 | 任意 | 任意 | `pytest path/to/file.py` |
| 任意命名 + 自动发现 | 任意 | 任意 | `pytest --ep-discover-all` |
| 混合使用 | 任意 | `test_*.py` | `pytest` |

**最简单的方式**：明确指定文件路径 `pytest your_file.py` ✨


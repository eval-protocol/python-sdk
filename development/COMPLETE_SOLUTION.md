# ✅ 完整解决方案：自动测试发现

## 🎯 问题和解决方案

### 问题
Pytest 对测试发现有严格的命名要求：
- 文件名必须是 `test_*.py` 或 `*_test.py`
- 函数名必须以 `test_` 开头

用户希望使用 `@evaluation_test` 装饰器后，无论如何命名都能被发现。

### 解决方案

#### ✅ 函数名：完全自动处理（无需任何操作）
使用 `@evaluation_test` 装饰的函数会自动注册正确的测试名称。

```python
# 任何函数名都可以！
@evaluation_test(...)
async def my_custom_eval(row: EvaluationRow) -> EvaluationRow:
    # 自动注册为 test_my_custom_eval
    ...
```

#### ✅ 文件名：三种方式

**方式 1：明确指定文件（最简单）**
```bash
pytest path/to/any_filename.py
```

**方式 2：使用标准命名**
```bash
# 文件名: test_*.py
pytest  # 自动发现
```

**方式 3：使用 --ep-discover-all 标志**
```bash
pytest --ep-discover-all  # 发现所有 .py 文件中的测试
```

## 📋 实现的代码修改

### 1. 函数名自动注册

**文件**: `eval_protocol/pytest/evaluation_test.py`

```python
# 在 decorator 返回之前自动注册
original_name = test_func.__name__
if not original_name.startswith('test_'):
    import sys
    frame = sys._getframe(1)
    caller_globals = frame.f_globals
    test_name = f'test_{original_name}'
    if test_name not in caller_globals:
        caller_globals[test_name] = dual_mode_wrapper
```

**工作原理**：
- 使用 `sys._getframe(1)` 获取调用者的全局命名空间
- 在命名空间中注册 `test_{function_name}` 别名
- Pytest 扫描模块时发现这个别名

### 2. Wrapper 名称修正

**文件**: `eval_protocol/pytest/parameterize.py`

```python
# 确保 wrapper 的 __name__ 以 test_ 开头
original_name = test_func.__name__
if not original_name.startswith('test_'):
    wrapper.__name__ = f'test_{original_name}'
```

**文件**: `eval_protocol/pytest/dual_mode_wrapper.py`

```python
# 确保 dual_mode_wrapper 的 __name__ 以 test_ 开头
original_name = test_func.__name__
if not original_name.startswith('test_'):
    dual_mode_wrapper.__name__ = f'test_{original_name}'
```

### 3. 文件名配置选项

**文件**: `eval_protocol/pytest/plugin.py`

```python
def pytest_addoption(parser) -> None:
    group = parser.getgroup("eval-protocol")
    group.addoption(
        "--ep-discover-all",
        action="store_true",
        default=False,
        help=(
            "Discover @evaluation_test in all Python files, "
            "not just test_*.py files."
        ),
    )

def pytest_configure(config) -> None:
    # 启用发现所有 .py 文件
    if config.getoption("--ep-discover-all", default=False):
        config.option.python_files = ["*.py"]
```

## 🧪 验证和测试

### 测试文件
- `tests/test_auto_discovery_simple.py` - 验证函数名自动注册
- `examples/auto_discovery_example.py` - 标准命名示例
- `examples/my_evaluation.py` - 非标准命名示例

### 验证结果

```bash
# 1. 非标准文件名 + 非标准函数名
$ pytest examples/my_evaluation.py --collect-only -v
collected 1 item
  <Coroutine test_custom_evaluation[rows(len=1)]>  ✅

# 2. 运行测试
$ pytest examples/my_evaluation.py -v
============================== 1 passed in 0.08s ===============================  ✅

# 3. 标准命名
$ pytest examples/auto_discovery_example.py --collect-only -v
collected 3 items
  <Coroutine test_math_evaluation[rows(len=1)]>              ✅
  <Coroutine test_greeting_evaluation[rows(len=1)]>          ✅
  <Coroutine test_coding_task_evaluation[rows(len=1)]>       ✅
```

## 📚 使用示例

### 示例 1：完全自由的命名

```python
# 文件: evals/math.py
from eval_protocol.pytest import evaluation_test
from eval_protocol.models import EvaluationRow, EvaluateResult

@evaluation_test(
    input_rows=[[EvaluationRow(messages=[{"role": "user", "content": "2+2"}])]]
)
async def evaluate_addition(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = EvaluateResult(score=1.0)
    return row
```

运行：
```bash
pytest evals/math.py -v
```

### 示例 2：使用标准命名

```python
# 文件: tests/test_math_eval.py
@evaluation_test(...)
async def test_addition(row: EvaluationRow) -> EvaluationRow:
    ...
```

运行：
```bash
pytest tests/  # 自动发现所有 test_*.py
```

### 示例 3：混合使用

```python
# 文件: test_my_evals.py（标准文件名）
@evaluation_test(...)
async def math_accuracy_check(row: EvaluationRow) -> EvaluationRow:
    # 函数名不标准也没问题
    ...
```

运行：
```bash
pytest  # 自动发现
```

## 🎁 特性总结

| 特性 | 状态 | 说明 |
|------|------|------|
| 函数名自由 | ✅ | 任何函数名都能被发现 |
| 文件名灵活 | ✅ | 支持明确指定或使用标志 |
| 零配置 | ✅ | 函数名完全自动处理 |
| 向后兼容 | ✅ | 不影响现有代码 |
| 无警告 | ✅ | 静默自动处理 |

## 📖 文档

- `development/auto_test_discovery.md` - 技术实现细节
- `development/file_and_function_naming.md` - 文件名和函数名处理指南
- `development/FINAL_SUMMARY.md` - 功能总结
- `development/COMPLETE_SOLUTION.md` - 本文档

## 🚀 推荐用法

### 最简单：明确指定文件
```bash
pytest path/to/your_file.py
```
- ✅ 任何文件名都可以
- ✅ 任何函数名都可以
- ✅ 无需额外配置

### 最传统：使用标准命名
```bash
# 文件: test_*.py
# 函数: test_* 或任意名称
pytest
```
- ✅ 自动发现
- ✅ 团队熟悉的方式

### 最灵活：使用 --ep-discover-all
```bash
pytest --ep-discover-all
```
- ✅ 发现所有文件中的测试
- ✅ 适合大量非标准命名文件

## ✨ 总结

现在使用 `@evaluation_test` 装饰器：

1. **函数名**：完全自由，自动处理 ✅
2. **文件名**：
   - 明确指定：`pytest your_file.py` ✅
   - 标准命名：`test_*.py` 自动发现 ✅
   - 或使用：`pytest --ep-discover-all` ✅

**用户只需要使用 `@evaluation_test`，其他都自动完成！** 🎉


# 自动测试发现功能 (Auto Test Discovery)

## 概述

`@evaluation_test` decorator 现在会自动确保所有装饰的函数都能被 pytest 发现，无论函数名是否遵循 pytest 命名规范。

## 功能说明

### 核心机制

当你使用 `@evaluation_test` 装饰一个函数时：

1. **如果函数名以 `test_` 开头**：正常工作，无需额外处理
2. **如果函数名不以 `test_` 开头**：decorator 会自动在模块的全局命名空间中注册一个以 `test_` 开头的别名

### 实现细节

- 在 `evaluation_test.py` 中，decorator 检查函数名
- 如果不以 `test_` 开头，使用 `sys._getframe(1).f_globals` 获取调用者的全局命名空间
- 在该命名空间中注册 `test_{original_name}` 别名
- Pytest 扫描模块时会发现这个别名

## 使用示例

```python
from eval_protocol.pytest import evaluation_test
from eval_protocol.models import EvaluationRow, EvaluateResult

# ✅ 这个函数名不以 test_ 开头，但仍然会被发现
@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[{"role": "user", "content": "Test"}])
    ]]
)
async def my_custom_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Pytest 会自动发现这个函数作为 'test_my_custom_evaluation'
    """
    row.evaluation_result = EvaluateResult(score=1.0)
    return row

# ✅ 这个函数名已经符合规范
@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[{"role": "user", "content": "Test"}])
    ]]
)
async def test_my_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    这个函数已经以 test_ 开头，正常工作
    """
    row.evaluation_result = EvaluateResult(score=1.0)
    return row
```

## 验证

运行 pytest collection 可以看到两个测试都被发现：

```bash
$ pytest --collect-only
<Module my_tests.py>
  <Coroutine test_my_custom_evaluation[rows(len=1)]>
  <Coroutine test_my_evaluation[rows(len=1)]>
```

## 代码修改

### 1. `eval_protocol/pytest/evaluation_test.py`

在 decorator 返回之前添加自动注册逻辑：

```python
# Auto-register the test function in the caller's namespace with 'test_' prefix
original_name = test_func.__name__
if not original_name.startswith('test_'):
    import sys
    frame = sys._getframe(1)
    caller_globals = frame.f_globals
    test_name = f'test_{original_name}'
    if test_name not in caller_globals:
        caller_globals[test_name] = dual_mode_wrapper
```

### 2. `eval_protocol/pytest/parameterize.py`

确保 wrapper 的 `__name__` 属性被修正：

```python
# Ensure wrapper name starts with 'test_' for pytest discovery
original_name = test_func.__name__
if not original_name.startswith('test_'):
    wrapper.__name__ = f'test_{original_name}'
```

### 3. `eval_protocol/pytest/dual_mode_wrapper.py`

同样确保 dual_mode_wrapper 的名称被修正：

```python
# Ensure the wrapper name starts with 'test_' for pytest discovery
original_name = test_func.__name__
if not original_name.startswith('test_'):
    dual_mode_wrapper.__name__ = f'test_{original_name}'
```

## 测试

参考 `tests/test_auto_discovery_simple.py` 查看完整的测试示例。

运行测试：
```bash
pytest tests/test_auto_discovery_simple.py -v
```

## 优点

1. ✅ 用户不需要记住命名规范
2. ✅ 所有使用 `@evaluation_test` 的函数都能被 pytest 发现
3. ✅ 无需任何配置
4. ✅ 向后兼容（已经使用 `test_` 前缀的函数继续正常工作）
5. ✅ 无警告，自动静默处理

## 总结

现在，只要你使用 `@evaluation_test` 装饰函数，就可以保证它能被 pytest 发现，无论你如何命名这个函数！🎉


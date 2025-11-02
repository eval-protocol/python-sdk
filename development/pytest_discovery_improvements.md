# Pytest Discovery Improvements

## 概述 (Overview)

为 `@evaluation_test` decorator 添加了自动验证功能，确保测试用例能够被 pytest 发现。

## 问题背景 (Background)

Pytest 对测试文件和函数的命名有严格要求：
- 测试文件必须命名为 `test_*.py` 或 `*_test.py`
- 测试函数必须以 `test_` 开头
- 测试类必须以 `Test` 开头

如果不遵循这些约定，pytest 将无法自动发现测试用例，导致测试无法运行。

## 实现的改进 (Improvements)

### 1. 函数名验证 (Function Name Validation)

**文件**: `eval_protocol/pytest/evaluation_test.py`

添加了 `_validate_pytest_discovery()` 函数，在装饰器应用时自动检查：
- ✅ 函数名是否以 `test_` 开头
- ✅ 文件名是否符合 `test_*.py` 或 `*_test.py` 模式

如果不符合规范，会发出清晰的警告信息，包含：
- 问题说明
- 修复建议
- 具体操作步骤

### 2. 自动名称修正 (Automatic Name Correction)

**文件**: `eval_protocol/pytest/parameterize.py`

在 `create_dynamically_parameterized_wrapper()` 函数中添加了自动修正逻辑：
- 如果原函数名不以 `test_` 开头，wrapper 函数名会自动添加 `test_` 前缀
- 这样即使原函数命名不规范，pytest 仍然能够发现测试

```python
# 原函数名: my_evaluation
# Wrapper 名: test_my_evaluation (自动修正)
```

### 3. 详细的警告信息 (Detailed Warning Messages)

警告信息格式化良好，易于阅读：

```
======================================================================
PYTEST DISCOVERY WARNING
======================================================================
Function 'my_evaluation' does not start with 'test_'.
Pytest will NOT discover this test automatically.

To fix this:
  1. Rename your function to 'test_my_evaluation', OR
  2. Run pytest with explicit path: pytest path/to/file.py::my_evaluation

Recommended: Rename to 'test_my_evaluation'
======================================================================
```

## 代码变更 (Code Changes)

### 1. `eval_protocol/pytest/evaluation_test.py`

- 添加 `import warnings`
- 新增 `_validate_pytest_discovery()` 函数
- 在 `decorator()` 函数中调用验证

### 2. `eval_protocol/pytest/parameterize.py`

- 修改 `create_dynamically_parameterized_wrapper()` 函数
- 添加自动名称修正逻辑

## 测试 (Tests)

创建了完整的测试套件：`tests/test_pytest_discovery_validation.py`

测试覆盖：
- ✅ 不规范命名时发出警告
- ✅ 规范命名时不发出警告
- ✅ Wrapper 名称自动修正
- ✅ 警告信息包含有用内容
- ✅ 与 pytest.mark.parametrize 兼容

所有测试通过！

## 文档 (Documentation)

### 1. 使用指南
**文件**: `docs/developer_guide/pytest_discovery_guide.mdx`

完整的文档，包括：
- Pytest 发现规则
- 最佳实践
- 故障排除
- 配置示例

### 2. 示例代码
**文件**: `examples/pytest_discovery_demo.py`

演示正确和错误的用法，以及如何使用新的验证功能。

## 使用示例 (Usage Examples)

### 正确用法 ✅

```python
from eval_protocol.pytest import evaluation_test
from eval_protocol.models import EvaluationRow, EvaluateResult

@evaluation_test(
    input_messages=[[{"role": "user", "content": "Hello"}]]
)
async def test_my_evaluation(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = EvaluateResult(score=1.0)
    return row
```

### 会触发警告但仍能工作 ⚠️

```python
@evaluation_test(
    input_messages=[[{"role": "user", "content": "Hello"}]]
)
async def my_evaluation(row: EvaluationRow) -> EvaluationRow:  # 警告：不以 test_ 开头
    row.evaluation_result = EvaluateResult(score=1.0)
    return row
```

虽然会警告，但 decorator 会自动修正 wrapper 名称，pytest 仍能发现此测试。

## 运行测试 (Running Tests)

```bash
# 运行所有测试
pytest

# 运行特定文件
pytest tests/test_evaluation.py

# 运行特定测试
pytest tests/test_evaluation.py::test_my_evaluation

# 查看哪些测试会被发现
pytest --collect-only
```

## 向后兼容性 (Backward Compatibility)

✅ **完全向后兼容**

- 不会破坏现有代码
- 仅添加验证和警告
- 自动修正确保测试仍然可以运行
- 所有现有测试继续正常工作

## 优势 (Benefits)

1. **早期发现问题**: 在定义测试时立即发现命名问题，而不是运行 pytest 时才发现
2. **清晰的指导**: 提供具体的修复建议和操作步骤
3. **自动修正**: 即使命名不规范，也能确保测试被发现
4. **更好的开发体验**: 减少因命名问题导致的调试时间

## 相关资源 (Resources)

- [Pytest Official Documentation](https://docs.pytest.org/en/stable/goodpractices.html#test-discovery)
- [Internal Documentation](../docs/developer_guide/pytest_discovery_guide.mdx)
- [Demo Example](../examples/pytest_discovery_demo.py)
- [Tests](../tests/test_pytest_discovery_validation.py)

## 总结 (Summary)

通过这些改进，`@evaluation_test` decorator 现在能够：

1. ✅ 自动验证命名约定
2. ✅ 提供清晰的警告和建议
3. ✅ 自动修正 wrapper 名称
4. ✅ 保持完全向后兼容
5. ✅ 提高开发者体验

开发者现在可以更自信地编写评估测试，知道如果有命名问题会立即得到反馈！


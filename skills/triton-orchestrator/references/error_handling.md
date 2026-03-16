# 错误处理策略

本文档定义了Triton算子生成系统的错误分类和处理策略。

---

## 1. 错误分类体系

### 1.1 编译错误 (Compilation Errors)

| 错误类型 | 特征 | 处理方式 |
|---------|------|---------|
| 简单编译错误 | 错误信息明确 + 修复方案清晰 | 自动修复 |
| 复杂编译错误 | 错误信息模糊/不明确 | 询问用户 |
| 语法错误 | Triton语法问题 | 自动修复 |
| 类型错误 | 数据类型不匹配 | 自动修复 |
| 形状错误 | Tensor形状不一致 | 分析后决定 |

**自动修复条件**：
- 错误信息包含明确的行号和错误描述
- 可以从知识库找到类似案例
- 修复方案不涉及架构变更

### 1.2 精度错误 (Precision Errors)

| 错误类型 | 特征 | 处理方式 |
|---------|------|---------|
| 简单精度错误 | 数值偏差小 + 原因明确 | 自动修复 |
| 复杂精度错误 | 数值偏差大/原因不明确 | 询问用户 |
| NaN/Inf错误 | 出现非数值 | 分析原因后决定 |
| 精度退化 | BF16→FP32转换问题 | 自动修复 |

**偏差阈值**：
- 相对误差 < 1e-3：可接受
- 相对误差 < 1e-2：需分析
- 相对误差 ≥ 1e-2：需询问用户

### 1.3 性能问题 (Performance Issues)

| 问题类型 | 特征 | 处理方式 |
|---------|------|---------|
| 性能不达标 | 精度正确但性能不达标 | 先尝试优化 |
| UB溢出 | 超出Unified Buffer容量 | 自动调整 |
| 流水线问题 | MTE/Vector未并行 | 自动优化 |
| 内存访问问题 | 访存效率低 | 自动优化 |

**处理策略**：
1. 先尝试自动优化（最多3次）
2. 优化失败后询问用户

### 1.4 硬件限制错误 (Hardware Constraints)

| 错误类型 | 特征 | 处理方式 |
|---------|------|---------|
| Block大小超限 | BLOCK_SIZE > 1024 | 询问用户 |
| 内存不足 | GM容量不足 | 询问用户 |
| 核数限制 | 超出可用核数 | 自动调整 |

---

## 2. 决策流程

```
错误发生
    │
    ▼
错误分类
    │
    ├─── 编译错误 ──┬─── 简单 ──→ 自动修复
    │               └─── 复杂 ──→ 询问用户
    │
    ├─── 精度错误 ──┬─── 偏差小 ──→ 自动修复
    │               └─── 偏差大 ──→ 询问用户
    │
    ├─── 性能问题 ──→ 尝试优化(3次) ──→ 失败 ──→ 询问用户
    │
    └─── 硬件限制 ──→ 询问用户
```

---

## 3. 自动修复策略

### 3.1 编译错误自动修复

```python
def auto_fix_compilation_error(code: str, error_msg: str) -> str:
    # 1. 从知识库检索类似案例
    similar_cases = knowledge_base.search_cases(error_msg)
    
    # 2. 分析错误原因
    error_type = classify_error(error_msg)
    
    # 3. 应用修复
    if error_type == "syntax":
        return fix_syntax_error(code, error_msg)
    elif error_type == "type":
        return fix_type_error(code, error_msg)
    elif error_type == "shape":
        return fix_shape_error(code, error_msg)
    
    return None  # 无法自动修复
```

### 3.2 精度错误自动修复

```python
def auto_fix_precision_error(code: str, error_info: dict) -> str:
    error_type = error_info.get("type")
    
    if error_type == "nan_inf":
        return add_numerical_stability(code)
    elif error_type == "precision_loss":
        return adjust_precision(code)
    elif error_type == "overflow":
        return add_overflow_protection(code)
    
    return None  # 无法自动修复
```

### 3.3 性能问题自动修复

```python
def auto_fix_performance(code: str, perf_info: dict) -> str:
    issues = analyze_performance_issues(code, perf_info)
    
    for issue in issues:
        if issue == "ub_overflow":
            code = adjust_block_size(code)
        elif issue == "pipeline":
            code = optimize_pipeline(code)
        elif issue == "memory_access":
            code = optimize_memory_access(code)
    
    return code
```

---

## 4. 询问用户策略

### 4.1 询问时机

- 自动修复失败
- 同类错误连续出现3次
- 涉及架构变更
- 需要用户确认方案

### 4.2 询问内容模板

**编译错误**：
```
遇到编译错误，无法自动修复：

错误信息：
{error_msg}

可能的原因：
{possible_causes}

建议的解决方案：
{suggested_solutions}

请选择：
1. 应用建议方案
2. 提供更多信息
3. 手动修改代码
```

**精度错误**：
```
精度验证失败：

相对误差：{relative_error}
绝对误差：{absolute_error}

问题位置：
{problem_location}

可能的原因：
{possible_causes}

请选择：
1. 接受当前精度
2. 尝试其他方案
3. 提供参考实现
```

**性能问题**：
```
性能优化尝试失败：

当前性能：{current_perf}
目标性能：{target_perf}

已尝试的优化：
{tried_optimizations}

请选择：
1. 继续优化
2. 接受当前性能
3. 调整性能目标
```

---

## 5. 重试策略

### 5.1 重试限制

| 类型 | 最大重试次数 | 连续失败阈值 |
|------|-------------|-------------|
| 编译错误 | 5次 | 3次同类错误 |
| 精度错误 | 5次 | 3次同类错误 |
| 性能优化 | 3次 | N/A |
| 总执行 | 30分钟 | N/A |

### 5.2 重试逻辑

```python
class RetryManager:
    def __init__(self):
        self.retry_count = 0
        self.error_history = []
        self.max_retries = 5
        self.same_error_threshold = 3
    
    def should_retry(self, error: Exception) -> bool:
        self.retry_count += 1
        
        if self.retry_count > self.max_retries:
            return False
        
        # 检查同类错误
        error_type = classify_error(error)
        same_errors = [e for e in self.error_history if classify_error(e) == error_type]
        
        if len(same_errors) >= self.same_error_threshold:
            return False
        
        self.error_history.append(error)
        return True
    
    def should_ask_user(self, error: Exception) -> bool:
        error_type = classify_error(error)
        same_errors = [e for e in self.error_history if classify_error(e) == error_type]
        return len(same_errors) >= self.same_error_threshold
```

---

## 6. 错误日志

所有错误都应记录到session的日志文件：

```json
{
  "timestamp": "2026-03-16T10:30:00",
  "error_type": "compilation",
  "error_message": "...",
  "stack_trace": "...",
  "auto_fix_attempted": true,
  "auto_fix_result": "failed",
  "user_action": "provided_more_info",
  "resolution": "..."
}
```

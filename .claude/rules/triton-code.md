---
paths:
  - "**/*.py"
  - "**/triton/**/*.py"
---

# Triton Code Rules

Triton代码规范，自动应用于所有Python文件。

## Kernel设计规范

- Block大小不超过1024
- 使用`tl.constexpr`声明编译期常量
- 避免在kernel中使用Python控制流
- 保持kernel函数简洁，单一职责

## 内存访问规范

- 优先连续内存访问
- 使用mask处理边界条件
- 避免离散访问模式
- 合并对同一地址的多次load

## 数值稳定性

- 使用float32进行中间计算
- 减最大值防止exp溢出
- 检查除零和NaN
- 避免大数相减

## 代码风格

- 函数命名：`snake_case`
- Kernel命名：`xxx_kernel`
- Host函数命名：`xxx`（与算子名一致）
- 添加必要的注释

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| UB溢出 | Block过大 | 减小BLOCK_SIZE |
| 精度损失 | BF16累加 | 使用float32累加 |
| 性能差 | 离散访问 | 优化内存模式 |
| 数值溢出 | exp(x)过大 | 减最大值 |
| NaN输出 | 除零或负数开方 | 添加保护 |

## Debugging

- 使用`tl.debug_barrier()`检查同步问题
- 打印中间结果调试数值问题
- 使用小规模数据验证正确性

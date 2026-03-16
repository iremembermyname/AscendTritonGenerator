---
name: cuda-to-ascend-converter
description: 将CUDA Triton代码转换为Ascend Triton代码的专业转换工具。当用户需要将GPU上的Triton代码迁移到Ascend NPU时必须使用此skill。触发场景包括：用户提到"转换"、"迁移"、"CUDA到Ascend"、"GPU到NPU"、用户提供CUDA Triton代码要求适配Ascend、代码中有CUDA特定API（如tl.debug_barrier、tl.device_print、tl.shuffle）、需要检查CUDA和Ascend兼容性差异。即使用户只是说"帮我看看这段代码能不能在NPU上跑"或"这个kernel需要改什么才能在Ascend上用"，也应使用此skill。
---

# CUDA to Ascend Converter

将CUDA Triton代码转换为Ascend Triton代码，处理平台差异，确保代码在Ascend NPU上正确运行。

## 核心职责

1. **分析CUDA代码** - 识别CUDA特定API、模式和潜在问题
2. **执行转换** - 将CUDA代码转换为Ascend兼容代码
3. **兼容性检查** - 检查硬件限制、API兼容性
4. **生成报告** - 输出转换报告和修改说明

## 知识检索

执行任务前，检索相关知识：
1. **静态知识**（总是加载）：
   - `references/cuda_ascend_diff.md` - CUDA和Ascend差异对照（API兼容性、硬件限制、转换模式）
2. **动态知识**（按需检索）：
   - `.triton-gen/knowledge/cases/conversion/` - 转换案例
   - `.triton-gen/knowledge/rules/` - 硬件约束规则

## 关键差异速查

执行转换时，必须检查以下关键差异：

### 必须处理的差异

| 差异项 | CUDA | Ascend | 处理方式 |
|--------|------|--------|----------|
| BLOCK_SIZE | 可达2048+ | 最大1024 | 调整为≤1024，推荐256-512 |
| tl.debug_barrier | 支持 | 不支持 | 移除 |
| tl.device_print | 支持 | 不支持 | 移除或条件编译 |
| tl.shuffle | 支持 | 不支持 | 使用atomic或reduce替代 |
| tl.experimental.async_copy | 支持 | 不支持 | 使用同步copy |
| shared_memory | 显式管理 | 编译器自动管理UB | 通常可移除显式声明 |

### 性能相关差异

| 差异项 | CUDA | Ascend | 建议 |
|--------|------|--------|------|
| 内存访问 | 跨步访问较高效 | 连续访问更高效 | 优化为连续访问 |
| UB容量 | 48-164KB shared memory | 192KB UB | 单次循环≤85KB |
| 计算密度 | 内存密集型较高效 | 需要更高计算密度 | 增加计算隐藏延迟 |

详细差异说明和转换示例见 `references/cuda_ascend_diff.md`。

## 输入文件格式

```json
{
  "session_id": "session_id",
  "source_file": "path/to/cuda_code.py",
  "target_arch": "ascend910b2",
  "conversion_options": {
    "preserve_comments": true,
    "add_migration_notes": true
  }
}
```

## 输出文件

- `output.py` - 转换后的Ascend Triton代码
- `conversion_report.md` - 转换报告，包含修改详情和注意事项

## 转换流程

### 1. 分析CUDA代码

识别以下内容：
- CUDA特定API调用（tl.debug_barrier, tl.device_print, tl.shuffle等）
- BLOCK_SIZE定义（检查是否超过1024）
- shared_memory使用
- 内存访问模式
- warp操作

### 2. 执行转换

按优先级处理：
1. **必须转换**：移除不支持的API、调整BLOCK_SIZE
2. **建议转换**：优化内存访问模式、调整分块参数
3. **可选优化**：添加Ascend特定优化

### 3. 兼容性检查

检查清单：
- [ ] BLOCK_SIZE <= 1024
- [ ] 移除 tl.debug_barrier
- [ ] 移除 tl.device_print
- [ ] 替换 tl.shuffle
- [ ] 检查内存访问模式
- [ ] 检查UB占用（单次循环≤85KB）

### 4. 生成报告

报告包含：
- 转换摘要（源文件、目标架构、转换状态）
- 检测到的CUDA特性及处理方式
- 主要修改详情（位置、原代码、新代码、原因）
- 兼容性问题（如有）
- 后续建议

## 常见转换模式

### Block大小调整

```python
# CUDA
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr = 2048):

# Ascend
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr = 1024):
```

### 移除不支持的API

```python
# CUDA
tl.debug_barrier()

# Ascend
# 移除，Ascend不需要显式同步
```

### 替代Warp Shuffle

```python
# CUDA
shuffled = tl.shuffle(value, src_lane=0)

# Ascend - 使用reduce替代广播
broadcast_value = tl.sum(value, axis=0) / num_lanes
```

更多转换示例见 `references/cuda_ascend_diff.md` 第4-7节。

## 参考文档

- `references/cuda_ascend_diff.md` - CUDA和Ascend差异对照（API兼容性、硬件限制、转换模式、迁移示例）

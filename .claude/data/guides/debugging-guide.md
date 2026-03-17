# Triton-Ascend 调试指南

本文档系统性地介绍 Triton-Ascend 编译与运行过程中常用的调试方法与工具。

---

## 目录

1. [编译流程概览](#1-编译流程概览)
2. [临时文件指引](#2-临时文件指引)
3. [解释器模式](#3-解释器模式)
4. [打印调试方法](#4-打印调试方法)
5. [编译错误调试方法](#5-编译错误调试方法)
6. [环境变量速查表](#6-环境变量速查表)

---

## 1. 编译流程概览

理解完整的编译链是进行有效调试的基础。Triton-Ascend 的编译过程遵循以下主要阶段：

| 阶段 | 输入 | 输出 | 工具/组件 | 说明 |
|:---|:---|:---|:---|:---|
| **Python Kernel编译** | `triton_kernel.py` | `ttir.mlir` | Triton JIT 编译器 | 将Triton Python kernel编译为标准Triton IR (TTIR) |
| **Triton IR 适配转换** | `ttir.mlir` | `ttadapter.mlir` | 适配Ascend的Triton后端 | **关键调试阶段**。将TTIR转换为面向Ascend NPU后端的适配器IR |
| **MLIR 编译与代码生成** | `ttadapter.mlir` | `.o` | 毕昇编译器 | 将适配器IR编译生成NPU可执行二进制代码 |

```bash
[Python Kernel]
     ↓ (triton.compile)
[ttir.mlir]
     ↓        │ (TRITON_DEBUG=1 → ~/.triton/dump/)
[ttadapter.mlir]
     ↓ (bishengir-compile)
[NPU 可执行文件 .o]
```

**调试重点**：`ttir.mlir` → `ttadapter.mlir` 的转换过程。

---

## 2. 临时文件指引

### 2.1 缓存文件（Cache）

缓存目录结构：
- 默认路径: `~/.triton/cache/`
- 主要缓存内容：ttir.mlir、ttadapter.mlir、编译产物

**缓存管理：**
```bash
# 清理缓存
rm -rf ~/.triton/cache/*

# 调试时禁用缓存
export TRITON_DISABLE_CACHE=1
```

### 2.2 调试转储文件（Dump Files）

通过设置 `TRITON_DEBUG=1`，可以在编译过程中将中间表示文件转储到磁盘。

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
python your_triton_program.py
```

转储目录：`~/.triton/dump/`

主要转储文件：
- `kernel.ttir.mlir`: Triton IR 文件（编译输入）
- `kernel.ttadapter.mlir`: 适配器 IR 文件（转换输出）

### 2.3 IR文件解析

#### TTIR（Triton Intermediate Representation）

TTIR 是 Triton 编译器前端生成的中间表示，保留了原始 Triton Python 内核的语义结构。

**示例：向量加法 TTIR**
```
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6, %cst : tensor<1024x!tt.ptr<f32>>
    ...
    tt.return
  }
}
```

**关键操作说明：**
- `tt.get_program_id`: 获取当前 block 的 ID
- `tt.make_range`: 构造 SIMD 风格的索引张量
- `tt.load`/`tt.store`: 向量化加载/存储
- `arith.cmpi`: 生成掩码防止越界

#### TTAdapter IR（Target-Specific Adapter Representation）

TTAdapter IR 是适配昇腾 NPU 架构的中间表示，采用标准 MLIR dialect（如 `memref`、`linalg`、`scf`）。

**示例：向量加法 TTAdapter IR**
```
module {
  func.func @add_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, ...) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %alloc = memref.alloc() : memref<1024xf32>
    ...
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %10 = arith.addf %8, %9 : tensor<1024xf32>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6
    return
  }
}
```

**关键转换：**
- 指针类型转换为 `memref`
- 引入边界检查逻辑（`scf.if`）
- 使用 `memref.alloc` 分配本地暂存 buffer
- 通过 `bufferization.to_tensor` 转为 tensor

---

## 3. 解释器模式

解释器的核心价值在于**隔离硬件差异**。通过环境变量 `TRITON_INTERPRET=1` 强制 Triton 在 CPU 上执行 kernel 计算，其结果可作为判断 NPU 计算精度的基准。

**使用方法：**

1. 设置环境变量并运行程序：
```bash
export TRITON_INTERPRET=1
python your_triton_program.py
```

2. 在 Triton kernel 源码中插入 Python 断点：
```python
@triton.jit
def my_kernel(...):
    x = tl.load(ptr + offsets)
    breakpoint()  # Python 内置断点函数
    y = compute(x)
```

3. 程序执行到断点会暂停，进入 Python 调试器 (`Pdb`)：
```python
(Pdb) p x  # 打印变量 x 的值
```

**注意**：解释器模式会在 CPU 上执行所有计算，显著降低运行效率。调试完成后务必取消设置：
```bash
unset TRITON_INTERPRET
# 或
export TRITON_INTERPRET=0
```

---

## 4. 打印调试方法

### 4.1 静态打印（tl.static_print）

在**编译时**打印常量表达式的值，适用于调试编译时已知的配置参数。

**特性：**
- 在编译时执行，不是运行时
- 只能打印编译时常量（`tl.constexpr` 参数、常量表达式）
- 输出显示在编译器的标准输出中

**使用方法：**
```python
@triton.jit
def triton_kernel(out_ptr, in_ptr, XBLOCK: tl.constexpr, USE_FP16: tl.constexpr):
    # 打印编译时常量参数
    tl.static_print("XBLOCK = ", XBLOCK)
    tl.static_print("USE_FP16 = ", USE_FP16)
    
    idx = tl.arange(0, XBLOCK)
    tmp = tl.load(in_ptr + idx)
    tl.store(out_ptr + idx, tmp)
```

```bash
export TRITON_DEVICE_PRINT=1
python your_program.py
```

### 4.2 运行时打印（tl.device_print）

在**运行时**打印张量值，是分阶段验证计算精度的高效方法。

**使用方法：**
```python
@triton.jit
def triton_kernel(out_ptr, in_ptr, XBLOCK: tl.constexpr):
    idx = tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr + idx)
    tmp1 = tmp0 * 2
    tl.device_print("tmp1 after multiply = ", tmp1)  # 打印中间结果
    tl.store(out_ptr + idx, tmp1)
```

```bash
export TRITON_DEVICE_PRINT=1
python your_program.py
```

**注意**：`tl.device_print` 在张量打印有长度限制，超长张量会被截断。

### 4.3 对比两种打印方法

| 特性 | `tl.device_print` | `tl.static_print` |
|------|-------------------|-------------------|
| **执行时机** | 运行时（kernel 执行时） | 编译时（kernel 编译时） |
| **输出位置** | 运行时标准输出 | 编译器标准输出 |
| **可打印内容** | 运行时张量值、变量 | 编译时常量、常量表达式 |
| **性能影响** | 有运行时开销 | 无运行时开销 |
| **启用环境变量** | `TRITON_DEVICE_PRINT=1` | `TRITON_DEVICE_PRINT=1` |

---

## 5. 编译错误调试方法

当 `ttir.mlir` → `ttadapter.mlir` 的转换过程失败，报错 `MLIRCompilationError` 时，需要进入代码层面定位问题。

### 5.1 Python 代码调试方法

使用 Python 内置的调试器 pdb 进行交互式调试：

```python
def compile_fn(ttir):
    import pdb; pdb.set_trace()  # 插入断点
    result = lower_function(ttir)
    return result
```

**pdb 常用命令：**
```python
(Pdb) l      # 查看当前代码上下文
(Pdb) p var  # 打印变量值
(Pdb) n      # 单步执行到下一行
(Pdb) c      # 继续执行
```

### 5.2 环境变量调试方法

#### MLIR_ENABLE_DUMP=1

启用 MLIR 高层 IR 的自动 dump，在每个 MLIR Pass 执行前后输出 IR 到 stderr。

```bash
export MLIR_ENABLE_DUMP=1
export TRITON_DEBUG=1
python your_triton_script.py
```

**特点**：日志量小，聚焦高层逻辑，日常调试首选。

#### TRITON_ENABLE_LLVM_DEBUG=1

启用 LLVM 后端 CodeGen 阶段的全量调试日志。

```bash
export TRITON_ENABLE_LLVM_DEBUG=1
export LLVM_DEBUG_ONLY="isel"  # 限制输出范围
python your_triton_script.py
```

**常用 DEBUG_TYPE：**
- `isel`：指令选择
- `regalloc`：寄存器分配
- `spiller`：寄存器溢出
- `peephole`：局部优化
- `asm-printer`：汇编生成

**调试流程推荐：**
1. 先启用 `MLIR_ENABLE_DUMP=1` 验证 MLIR 层转换
2. 若 MLIR 正常但结果错误，再启用 `TRITON_ENABLE_LLVM_DEBUG=1`

---

## 6. 环境变量速查表

| 变量 | 作用 |
|------|------|
| `TRITON_DEBUG=1` | 启用中间 IR 转储 |
| `TRITON_DISABLE_CACHE=1` | 禁用编译缓存 |
| `TRITON_INTERPRET=1` | 使用 CPU 解释器执行 kernel |
| `TRITON_DEVICE_PRINT=1` | 启用运行时打印输出 |
| `MLIR_ENABLE_DUMP=1` | 启用 MLIR 高层 IR dump |
| `TRITON_ENABLE_LLVM_DEBUG=1` | 启用 LLVM 后端调试日志 |
| `LLVM_DEBUG_ONLY="xxx"` | 限制 LLVM 调试输出范围 |

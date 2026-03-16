# 任务分类规则

本文档定义了Triton算子生成系统的任务分类标准和判断规则。

---

## 1. 任务类型定义

### 1.1 算子生成 (Operator Generation)

**特征关键词**：
- "生成"、"创建"、"写一个"、"实现"
- "算子"、"kernel"、"op"
- 具体算子名称：softmax、layernorm、matmul、attention等

**判断规则**：
```python
def is_generation_task(query: str) -> bool:
    generation_keywords = ["生成", "创建", "写一个", "实现", "generate", "create", "implement"]
    op_keywords = ["算子", "kernel", "op", "operator"]
    
    has_generation = any(kw in query.lower() for kw in generation_keywords)
    has_op = any(kw in query.lower() for kw in op_keywords)
    
    return has_generation and has_op
```

**示例**：
- "帮我生成一个融合softmax算子"
- "写一个layernorm kernel"
- "实现一个matmul算子"

---

### 1.2 代码优化 (Code Optimization)

**特征关键词**：
- "优化"、"提升"、"加速"、"改进"
- "性能"、"速度"、"效率"
- "慢"、"太慢"、"不够快"

**判断规则**：
```python
def is_optimization_task(query: str) -> bool:
    opt_keywords = ["优化", "提升", "加速", "改进", "optimize", "improve", "speed up"]
    perf_keywords = ["性能", "速度", "效率", "performance", "speed", "slow"]
    
    has_opt = any(kw in query.lower() for kw in opt_keywords)
    has_perf = any(kw in query.lower() for kw in perf_keywords)
    
    return has_opt or has_perf
```

**示例**：
- "优化这段代码的性能"
- "这个kernel太慢了，帮我加速"
- "提升matmul算子的效率"

---

### 1.3 代码转换 (Code Conversion)

**特征关键词**：
- "转换"、"迁移"、"移植"
- "CUDA"、"GPU" → "Ascend"、"NPU"
- "适配"、"兼容"

**判断规则**：
```python
def is_conversion_task(query: str) -> bool:
    conv_keywords = ["转换", "迁移", "移植", "convert", "migrate", "port"]
    source_keywords = ["cuda", "gpu", "nvidia"]
    target_keywords = ["ascend", "npu", "华为"]
    
    has_conv = any(kw in query.lower() for kw in conv_keywords)
    has_source = any(kw in query.lower() for kw in source_keywords)
    has_target = any(kw in query.lower() for kw in target_keywords)
    
    return has_conv and (has_source or has_target)
```

**示例**：
- "把这个CUDA kernel转换成Ascend版本"
- "迁移这段GPU代码到NPU"
- "把CUDA Triton代码移植到Ascend"

---

### 1.4 知识添加 (Knowledge Addition)

**特征关键词**：
- "发现问题"、"找到原因"
- "添加知识"、"记录经验"
- "这个坑"、"踩坑"、"避坑"

**判断规则**：
```python
def is_knowledge_task(query: str) -> bool:
    knowledge_keywords = ["添加知识", "记录", "发现", "经验", "坑", "问题"]
    action_keywords = ["保存", "存储", "记住", "记录"]
    
    has_knowledge = any(kw in query.lower() for kw in knowledge_keywords)
    has_action = any(kw in query.lower() for kw in action_keywords)
    
    return has_knowledge and has_action
```

**示例**：
- "发现了一个问题，帮我记录到知识库"
- "这个坑要记住，添加到知识库"

---

### 1.5 问题诊断 (Problem Diagnosis)

**特征关键词**：
- "报错"、"错误"、"异常"
- "为什么"、"什么原因"
- "调试"、"排查"、"定位"

**判断规则**：
```python
def is_diagnosis_task(query: str) -> bool:
    error_keywords = ["报错", "错误", "异常", "error", "exception", "fail"]
    question_keywords = ["为什么", "什么原因", "怎么解决", "why", "how"]
    debug_keywords = ["调试", "排查", "定位", "debug", "diagnose"]
    
    has_error = any(kw in query.lower() for kw in error_keywords)
    has_question = any(kw in query.lower() for kw in question_keywords)
    has_debug = any(kw in query.lower() for kw in debug_keywords)
    
    return has_error or has_question or has_debug
```

**示例**：
- "为什么这个kernel报错了"
- "帮我调试一下这个错误"
- "这个异常是什么原因"

---

## 2. 任务优先级

当任务可能属于多个类型时，按以下优先级判断：

1. **知识添加** - 最高优先级，用户明确要保存知识
2. **问题诊断** - 需要先解决问题才能继续
3. **代码转换** - 转换后可能需要验证和优化
4. **算子生成** - 标准生成流程
5. **代码优化** - 可能在生成或转换后进行

---

## 3. 复合任务处理

某些任务可能包含多个阶段：

**示例**： "帮我生成一个softmax算子，并优化性能"

处理方式：
1. 先识别为复合任务
2. 按顺序执行：生成 → 验证 → 优化
3. 每个阶段独立调用相应skill

---

## 4. 任务信息提取

对于每种任务类型，需要提取的关键信息：

### 算子生成
- 算子名称
- 输入/输出规格
- 数据类型
- 性能要求

### 代码优化
- 目标代码文件
- 当前性能数据
- 目标性能指标

### 代码转换
- 源代码文件
- 源平台（CUDA/GPU）
- 目标平台（Ascend/NPU）

### 知识添加
- 知识类型（案例/规则/文档）
- 知识内容
- 相关标签

### 问题诊断
- 错误信息
- 相关代码
- 运行环境

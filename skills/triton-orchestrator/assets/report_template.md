# Triton算子生成报告模板

## 任务信息

- **Session ID**: {session_id}
- **任务类型**: {task_type}
- **创建时间**: {created_at}
- **完成时间**: {completed_at}

## 执行摘要

{summary}

## 生成的代码

### 文件路径
- 主文件: `{main_file_path}`
- 测试文件: `{test_file_path}` (如有)

### 代码预览

```python
{code_preview}
```

## 验证结果

### 精度验证
| 指标 | 值 |
|------|-----|
| 最大相对误差 | {max_relative_error} |
| 平均相对误差 | {avg_relative_error} |
| 验证状态 | {verification_status} |

### 性能数据 (如有)
| 指标 | 值 |
|------|-----|
| 执行时间 | {execution_time} us |
| 内存占用 | {memory_usage} |
| 性能状态 | {performance_status} |

## 优化记录 (如有)

### 应用的优化技术
{optimization_techniques}

### 优化前后对比
| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 执行时间 | {before_time} us | {after_time} us | {speedup}x |

## 问题与解决

{issues_and_solutions}

## 知识更新 (如有)

{knowledge_updates}

## 下一步建议

{next_steps}

---

*报告生成时间: {report_time}*

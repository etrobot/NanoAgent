### NanoAgent

NanoAgent 是一个基于 Python 的框架，用于构建逻辑助理Agent，能够通过逐步规划和执行来解决用户请求。它集成了 OpenAI API 以实现自然语言处理，并提供结构化的基于动作的工作流程，来生成智能响应。

```mermaid
graph TD
    A[开始] --> B[接收任务]
    B --> C[大模型回答]
    C --> D[解析下一步动作]
    D --> E{是否结束答案？}
    E -- 是 --> F[结束并输出]
    E -- 否 --> G[工具调用]
    G --> B
```
---

## 功能特点

- **逐步推理：** 系统化地处理用户查询，采用规划和行动执行的方法。
- **自定义动作：** 可根据具体使用场景定义额外的自定义动作。

---

## 安装

通过以下方式克隆代码库并安装依赖：

```bash
pip install git+https://github.com/etrobot/nanoagent.git
```
或使用 Poetry：
```bash
poetry add git+https://github.com/etrobot/nanoagent.git
```

---

## 使用方法

### 初始化

使用您的 OpenAI API 密钥和所需配置来初始化 NanoAgent：

```python
from nanoagent import NanoAgent

agent = NanoAgent(
    api_key="your_openai_api_key",
    base_url="your_base_url", 
    model="your_model", 
    max_tokens=your_max_tokens, 
    actions=["custom_action"], 
    debug=True
)
```

参数说明：
- `api_key`：您的 OpenAI API 密钥。
- `base_url`：OpenAI API 的端点。
- `model`：所使用的 OpenAI 模型（例如 `gpt-4`）。
- `max_tokens`：模型生成响应的最大 Token 数。
- `actions`：自定义动作的列表（默认包含 `think_more` 和 `end_answer`）。
- `debug`：启用日志记录，便于调试。

---

### 运行Agent

使用 `run()` 方法处理用户查询，Agent会进行规划、执行和响应。

```python
agent.run("法国的首都是哪里？")
```

---

### 自定义工具

通过定义自定义工具扩展功能。例如：

```python
def custom_action(input):
    return f"执行了自定义工具，输入内容是：{input}"
```

在初始化时，将自定义工具添加到 `actions` 列表中。

---

### 调试与日志

初始化时设置 `debug=True` 可启用详细日志，记录交互和动作执行的详细信息。

```python
agent = NanoAgent(api_key="your_key", base_url="url", model="gpt-4o-mini", max_tokens=1024, debug=True)
```

日志内容包括用户查询、助手响应、选定的动作及其理由。

---

## 示例工作流

1. **用户查询：** 用户提供查询，例如 *"解释光合作用的工作原理。"*
2. **Agent响应：** 助手生成详细响应，并选择一个定义的动作。
3. **动作执行：** 执行选定动作，并将结果集成到工作流中。
4. **最终输出：** Agent输出最终答案或根据需要继续工作流。

---

## 许可证

本项目基于 MIT 许可证开源发布。详情请参阅 [LICENSE](LICENSE) 文件。
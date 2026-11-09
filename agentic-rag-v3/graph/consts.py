RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"
GIVE_UP = "give_up"

# 自纠正重试上限：证据不支持时最多重生成 MAX_ITERATIONS 次，之后明确拒答。
# 用状态计数而不是"消息条数阈值"，避免多轮工具调用时过早终止或无限循环。
MAX_ITERATIONS = 2

# 交付范围与安全边界（2026-09-30）

用户最新决定：不部署 Docker，先交付本机项目。
不再修复 Docker Desktop，不删除损坏 socket、镜像、容器或数据卷。
保留配置不等于容器部署已验收。

## 本机交付

- 统一入口 python -m app.main；前端生产构建通过该 API 同源提供。
- scripts/start_local.ps1 -Port 8001 使用回环地址、隐藏后台窗口与独立日志，不重复占用端口。
- .env、JWT、管理员口令、Ark/Tavily Key 不提交，不输出到公开日志。
- 登录、管理员接口与用户所属会话/run/证据/工单隔离有自动测试，上传有文件边界校验。
- 当前知识快照为登录用户共享，不是多租户私有文档权限平台。
- 保持内存、有限等待槽、总超时、预算、取消和未知结果熔断，不绕过保护来演示。
- 备份脚本校验三类 SQLite 一致性备份与隔离恢复；知识、图片、模型另行备份。
- 代码回退使用独立目录和对应快照，不覆盖用户已有工作，不倒退预算账本。

## 可选容器方案（未做运行验收）

Dockerfile 多阶段构建 Vue 与 Python，以非 root 启动；
docker-compose.yml 使用 PostgreSQL 16、PostgresSaver 和持久卷，
API 默认只发布到 127.0.0.1:8002。
requirements-linux.lock 固定 Linux CPU 依赖与包摘要。
scripts/init_container_env.py 用随机值创建忽略的 .env.container，拒绝覆盖已有凭据。
配置静态校验通过；实际镜像构建、启动、数据库迁移与容器恢复未执行。

将来明确恢复容器工作后才执行：

    .\.venv\Scripts\python.exe scripts/init_container_env.py
    docker compose --env-file .env.container config --quiet
    docker compose --env-file .env.container up --build -d

已有 .env.container 时跳过初始化，安全管理其密码。
需要 output/ 与同级 model_cache/huggingface；镜像不包含原始语料和模型，不是独立完整数据交付。
容器默认关闭付费调用与视觉塔，必须按资源条件重新配置和验收，不能称为全通道上线。
首次启动只为新 PostgreSQL 登记当前知识快照，不迁移本机历史会话，不覆盖旧活动快照。

生产启动配置拒绝弱 JWT/管理员密码、相同凭据、通配 CORS、调试模式和未批准算法；
开发服务拒绝绑定非回环地址。安全配置验证不等于生产运行验收。
公网部署不在本轮范围；未来需目标机实测、HTTPS 反向代理、访问限制、密钥轮换、
监控、备份轮换和经人工审核的质量门禁。
禁止以测试全绿、HTTP 200 或模型自判通过代替金融答案正确性与上线批准。

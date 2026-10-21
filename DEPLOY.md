# 部署手册 — 把 aml-memory 挂到公网并接上 AML 评测平台

你只需要一台有公网 IP 的 Linux 机器和一个域名。全过程约 20 分钟，之后平台就能用
`https://<你的域名>/add`、`/search`、`/health` 调你的服务。

> 为什么必须自己部署：Cycle 2 的规则明确写了「只交 GitHub 仓库 / 只交 Docker 镜像 /
> 让平台帮你启动」都会被拒，**必须提供一个活着的、可公网访问的 Add/Search 端点**。

---

## 0. 你需要准备什么

| 东西 | 说明 |
|---|---|
| 一台 VPS | 2 vCPU / 2 GB 内存 / 20 GB 磁盘起步（4 GB 更稳）。阿里云/腾讯云/轻量应用服务器都行，**需要有公网 IPv4** |
| 一个域名 | 任意注册商。解析一条 A 记录到 VPS 的公网 IP，例如 `memory.example.com → 1.2.3.4` |
| 80 / 443 端口 | 安全组 / 防火墙放行 TCP 80 和 443（Caddy 申请证书必须走 80） |
| DashScope Key | 已有的 `sk-...`，用于 `text-embedding-v4` 和 `gte-rerank-v2` |
| 一个自定的 API Key | 平台调用你时用的 Bearer token，自己随便定一个长随机串（例如 `openssl rand -hex 24` 生成） |

域名解析生效后再往下做（`ping memory.example.com` 能出你的公网 IP 即可），否则 Caddy
申请证书会失败。

---

## 1. 装 Docker

```bash
# Debian / Ubuntu
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker
docker --version && docker compose version
```

> 国内机器拉镜像慢可在 `/etc/docker/daemon.json` 配镜像加速后 `systemctl restart docker`。

## 2. 上传代码

在本地把整个 `aml-memory/` 目录传上去（或用 git）：

```bash
# 本机执行
scp -r aml-memory user@1.2.3.4:~/aml-memory
# 或者：rsync -av aml-memory/ user@1.2.3.4:~/aml-memory/
```

目录里应该有：`aml_memory/`、`deploy/`、`Dockerfile`、`requirements.txt`、`.env.example`。

## 3. 配置 `.env`

```bash
cd ~/aml-memory
cp .env.example .env
nano .env
```

必须改的 6 行：

```ini
# 平台调用你时用的 token —— 自己定，等下注册时填同一个
AML_API_KEY=换成你自己的长随机串

# 你自己的域名；compose 用它给 Caddy 申请证书
AML_DOMAIN=memory.example.com

# DashScope
EMBED_PROVIDER=openai
EMBED_MODEL=text-embedding-v4
EMBED_DIM=1024
EMBED_BASE_URL=https://ws-i35enzeci2mcdfzq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
EMBED_API_KEY=你的 DashScope key

# 重排（可选但推荐；不想要就写 none）
RERANK_PROVIDER=dashscope
RERANK_MODEL=gte-rerank-v2
RERANK_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_API_KEY=你的 DashScope key
```

其余项保持默认即可（默认值就是评测下来最好的一组，见 README 第 3 节）。
`AML_DATA_DIR` **不要改**，compose 会强制成 `/data`（挂载卷）。

## 4. 起服务

```bash
cd ~/aml-memory/deploy
docker compose up -d --build
docker compose ps          # 两个容器都应为 running / healthy
docker compose logs -f aml-memory
```

首次启动约 1–2 分钟（装依赖 + 申请证书）。日志里出现
`aml-memory 1.x starting: add=/add search=/search health=/health embed=openai/text-embedding-v4`
即成功。

## 5. 自测（务必先自己跑通再交给平台）

```bash
# 1) 健康检查：不需要 token
curl -s https://memory.example.com/health
# → {"status":"ok","service":"aml-memory","version":"..."}

# 2) 写入
curl -s -X POST https://memory.example.com/add \
  -H "Authorization: Bearer 你的AML_API_KEY" -H 'content-type: application/json' \
  -d '{"request_id":"smoke-1","user_id":"u1","session_id":"s1",
       "messages":[{"role":"user","content":"Caroline: I adopted a greyhound named Pepper in May 2023.","timestamp":1683093600000},
                   {"role":"assistant","content":"Melanie: Lovely! I signed up for a pottery class yesterday."}]}'
# → {"success":true,"request_id":"smoke-1","user_id":"u1","session_id":"s1"}

# 3) 立刻检索（必须能搜到刚写的内容）
curl -s -X POST https://memory.example.com/search \
  -H "Authorization: Bearer 你的AML_API_KEY" -H 'content-type: application/json' \
  -d '{"query":"What is the name of Caroline'"'"'s dog?","user_id":"u1","top_k":10}'
# → {"data":[{"id":"...","content":"Caroline (8 May 2023): I adopted a greyhound named Pepper ...","score":...}]}

# 4) 鉴权是否真的生效（应返回 401）
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://memory.example.com/search \
  -H 'content-type: application/json' -d '{"query":"x","user_id":"u1","top_k":5}'
```

四条都对，就可以去平台登记了。

更省事的办法：用仓库自带的一次性契约自检脚本（覆盖健康检查、鉴权、写入即可见、
`request_id` 重放、`top_k` 截断、未知用户空列表、错误请求必须 4xx、检索延迟）：

```bash
# 在 VPS 上（或在任何能访问该域名的机器上）
python3 -m pip install httpx
python3 scripts/verify_endpoint.py --base-url https://memory.example.com \
    --token '你的AML_API_KEY'
# → PASS/FAIL 表格；最后一行 "all 15 checks passed" 才算过
```

注意脚本会用一个随机 `user_id`（`probe-xxxxxxxx`）写几条测试数据，讲完就留在库里，
不影响正式评测（`user_id` 是唯一的隔离边界）；介意的话 Full 前删掉卷再重建也行，
但重建会丢掉已写入的历史——正常情况下不必。

## 6. 在 AML 平台登记

在 Agent Memory Leaderboard 的提交/评测页面填：

| 字段 | 填什么 |
|---|---|
| Add URL | `https://memory.example.com/add` |
| Search URL | `https://memory.example.com/search` |
| Health URL | `https://memory.example.com/health`（可选，建议填） |
| Auth | `Bearer` + 你 `AML_API_KEY` 的值（或 `X-Api-Key` 头，两者都支持） |
| Division | Open-source Methods（方法开源才能拿奖） |
| Embedding model | `text-embedding-v4` |
| LLM in Add/Search | `N/A`（我们不用生成模型） |
| Reranker | `gte-rerank-v2` |

先跑 **Smoke**（每小时 1 次、每 track 最多 30 次），确认 schema 全过；再排 **Full**。
Full 期间服务版本会被冻结，别改代码、别升级依赖、别动 `.env` 里的检索参数。

## 7. 运维要点

* **数据在命名卷 `aml-data` 里**（容器内 `/data/aml.sqlite3`）。备份：
  ```bash
  docker run --rm -v deploy_aml-data:/data -v $PWD:/backup alpine \
    tar czf /backup/aml-$(date +%F).tgz -C /data .
  ```
* **重启安全**：`request_id` 幂等记录在 SQLite 里，容器重启后重放同一个 `request_id`
  不会重复写入（`tests/test_durability.py` 覆盖了这条）。Full 期间允许无状态重启。
* **超时**：平台单次 Add 最长 30 分钟，Caddy 已配 1900 s 读写超时；不要在前面再加一层
  默认 60 s 超时的 Nginx。
* **并发**：单进程 uvicorn + 线程池，`AML_MAX_CONCURRENT_SEARCH=16`、`_INGEST=8`。
  需要更高就调这两个值，或把机器升到 4 vCPU。
* **日志**：`docker compose logs -f aml-memory`；已限制单文件 50 MB × 5。
* **健康检查**：Caddy 依赖 `service_healthy` 才启动，健康检查失败时不会对外提供服务，
  避免把 502 暴露给评测。

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `caddy` 反复重启，日志 `obtaining certificate: ...` | 域名没解析到本机，或 80 端口没放行。`dig +short memory.example.com` 确认 |
| `dial tcp: lookup aml-memory` | compose 没起来，先在 `deploy/` 目录执行（不要在别的目录 `-f` 指来指去） |
| `/search` 返回 401 | 平台给的 token 和你 `.env` 里的 `AML_API_KEY` 不一致；或误加了空格 |
| 502 Bad Gateway | `aml-memory` 容器不健康：`docker compose logs aml-memory` 看 embedding key 是否写错 |
| embedding 报 400/401 | `EMBED_BASE_URL` 要保留 `/compatible-mode/v1` 结尾；key 与 base 要属于同一个 DashScope 账号 |
| 写入很慢 | 每次 Add 最多 20 条消息，正常 < 3 s；若持续 30 s+ 看 `EMBED_CONCURRENCY` 和网络 |
| 想换机器 | 备份 `aml-data` 卷 → 新机恢复 → 改 DNS → 重新起。**Full 期间不要换** |

## 9. 本地先验一遍（可选）

同一套代码在本机跑不需要 Docker：

```bash
cd aml-memory
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env      # 填 EMBED_API_KEY，AML_API_KEY 留空 = 本机免鉴权
python -m uvicorn aml_memory.main:app --port 8000
python -m pytest tests -q  # 28 个用例，全离线（用 hash embedder），不需要 key
```

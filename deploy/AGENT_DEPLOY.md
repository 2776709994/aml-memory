# 服务器部署作业单（给运维 Agent）

你在一台 Linux 服务器上，要把它变成"AML 记忆服务"的对外服务节点。**照单执行，不要自由发挥**。
全部工作由仓库里的 `deploy/bootstrap.sh` 完成；本文告诉你需要什么输入、怎么判断成功、什么时候必须停下。

---

## 0. 一句话目标

把 <https://github.com/2776709994/aml-memory> 的 `v1.2` 标签部署到本机，
对外提供 **HTTPS** 的 `/add`、`/search`、`/health`，通过 13 项契约自检，然后**停下**等人类做榜单注册。

---

## 1. 开工前，向人类确认三件事（缺任何一项 → 停下并提问，不要猜）

| 输入 | 说明 | 示例 |
|---|---|---|
| `AML_DOMAIN` | 已解析到**本机公网 IP** 的域名（A 记录）。Caddy 会自动申请证书，所以 80/443 必须能从公网访问 | `mem.example.com` |
| `EMBED_API_KEY` | 能调用 `text-embedding-v4` 的 DashScope API Key（`sk-` 开头） | `sk-...` |
| 权限 | 你是否能用 `sudo`（脚本需要 root 安装 Docker、写 `/opt`、绑定 80/443） | — |

如果你拿不到上面任一项，**不要用占位符硬跑**，直接回报缺什么。

---

## 2. 环境自检（1 分钟，只读，不改任何东西）

```bash
whoami; sudo -n true && echo "sudo ok"
nproc; free -m | head -2; df -h /var/lib | tail -1
getent ahostsv4 mem.example.com | head -1        # 必须返回一个 IP
curl -fsS https://api.ipify.org; echo            # 本机公网 IP，应与上面一致
ss -ltn | grep -E ':(80|443)\b' || echo "80/443 free"
command -v docker && docker --version; command -v git; command -v openssl
```

判断标准：

* 内存 ≥ 2 GB、`/var/lib` 剩余 ≥ 8 GB；
* 域名解析到的 IP == `api.ipify.org` 返回的 IP（不一致 → 证书一定签不下来，先让人类改 DNS）；
* 80/443 没有别的进程占用（有 nginx/apache 就先停掉：`systemctl disable --now nginx`）；
* 没有 Docker 也可以，脚本会自己装（走阿里云镜像源，约 2 分钟）。

---

## 3. 执行部署（一条命令，可重复执行）

```bash
sudo AML_DOMAIN=mem.example.com \
     EMBED_API_KEY=sk-你的key \
     bash <(curl -fsSL https://raw.githubusercontent.com/2776709994/aml-memory/v1.2/deploy/bootstrap.sh)
```

如果 GitHub 直连不通（`curl` 超时），用下面两种方式之一，效果相同：

```bash
# A. 先克隆仓库，再从本地文件执行
git clone https://github.com/2776709994/aml-memory.git /opt/aml-memory-src
sudo AML_DOMAIN=... EMBED_API_KEY=... bash /opt/aml-memory-src/deploy/bootstrap.sh

# B. 人类给你一个 tar 包（aml-memory-v1.2.tar.gz）时
mkdir -p /opt/aml-memory-src && tar xzf aml-memory-v1.2.tar.gz -C /opt/aml-memory-src --strip-components=1
sudo AML_DOMAIN=... EMBED_API_KEY=... AML_SRC=/opt/aml-memory-src bash /opt/aml-memory-src/deploy/bootstrap.sh
```

脚本做的事（按顺序，全部自带检查）：

1. 预检：root、内存、磁盘、DNS、80/443、必要命令；
2. 安装并启动 Docker（已装则跳过）；
3. `git clone` 到 `/opt/aml-memory` 并 `checkout v1.2`（已存在则原地更新）；
4. 生成 `/opt/aml-memory/.env`（`AML_API_KEY` 用 `openssl rand` 生成，`chmod 600`）；**文件已存在时不会覆盖，token 保持不变**；
5. `docker compose --env-file ../.env up -d --build`（aml-memory + caddy 两个容器）；
6. 等容器 `healthy`（最多 5 分钟）；
7. 在容器内跑 13 项契约自检；
8. 等 `https://<域名>/health` 返回 200（首次签证书需要几秒到一分钟）；
9. 打印部署报告（含要填进榜单的 URL 与 token）。

预计耗时：首次 5–10 分钟（含镜像构建），重复执行 1–2 分钟。

可选参数：`AML_TAG`（默认 `v1.2`）、`AML_DIR`（默认 `/opt/aml-memory`）、`AML_TOKEN`（固定 token）、`SKIP_DNS_CHECK=1`（DNS 还没生效但想先把服务跑起来）、`SKIP_VERIFY=1`。

---

## 4. 判断成功（必须看到这三样）

1. 脚本最后一行是 `Deployment complete.`；
2. 自检段落里每一项都是 `PASS`，末尾类似 `13/13 checks passed`；
3. 下面三条命令的结果与期望一致：

```bash
cd /opt/aml-memory/deploy
docker compose --env-file ../.env ps                     # aml-memory: healthy, aml-caddy: running
curl -s -o /dev/null -w '%{http_code}\n' https://mem.example.com/health          # 200
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://mem.example.com/search \
     -H 'content-type: application/json' -d '{"query":"x","user_id":"y"}'        # 401（没带 token 被拒）
```

再跑一次自检（任何时候都可以，不会破坏数据）：

```bash
TOKEN=$(grep '^AML_API_KEY=' /opt/aml-memory/.env | cut -d= -f2)
docker compose --env-file ../.env exec -T aml-memory \
  python scripts/verify_endpoint.py --base-url http://127.0.0.1:8000 --token "$TOKEN"
```

---

## 5. 到此为止，剩下的事必须由人类做

| 事项 | 为什么 Agent 不能做 |
|---|---|
| 在 agentmemoryleaderboard.ai 注册账号 / 申请评测 key | 需要邮箱验证、OAuth 登录、以及人类的比赛账号 |
| 填写 Add/Search/Health URL 与 token | 属于报名信息，需要人类确认 |
| 跑 Smoke（每小时 ≤1 次，每题 ≤30 次）→ 再跑 Full | Full 每 key 每赛道只有 2 次机会，且必须 100% 完成，跑错就作废 |

**把第 4 节的三条验证结果 + 部署报告（URL、token、tag、commit）回报给人类，然后停止。**

---

## 6. 禁止事项（会直接毁掉评测）

1. **不要**修改 `/opt/aml-memory/aml_memory/` 下的任何代码，也不要改 `.env` 里除域名/token/key 之外的调参项（`AML_CONTEXT_*`、`AML_CANDIDATE_DEPTH`、`AML_RRF_K`、`RERANK_*`、`EMBED_*` 的模型名等）——这些值是测过的，改了分数会掉；
2. **不要**把 8000 端口暴露到公网（compose 里只有 caddy 映射 80/443，服务只在内部网络）；
3. **不要**设置 `AML_DISABLE_AUTH=1`，也不要把 token 改成空 —— 空 token 等于任何人都能写库；
4. **不要**删除 `aml-data` 卷（`docker volume rm`）——里面是记忆库；
5. **不要**自己发起 AML 的 Smoke / Full 评测；
6. **不要**把 `.env` 或 token 提交到 git、贴到公开聊天、写进日志；
7. **不要**升级系统大版本或重启机器而不检查 `docker compose ps`；机器重启后容器会自启动（`restart: unless-stopped`）。

---

## 7. 故障对照表

| 现象 | 原因 | 处置 |
|---|---|---|
| `port 80 is already in use` | 已有 nginx/apache | `systemctl disable --now nginx apache2` 后重跑 |
| `DNS: <域名> does not resolve` | A 记录没生效 | 等 5–30 分钟或让人类改 DNS；临时可 `SKIP_DNS_CHECK=1`（HTTPS 会失败，服务本身可用） |
| `https://<域名>/health` 非 200，但容器 healthy | 证书没签下来 | `docker compose --env-file ../.env logs --tail 50 caddy`；常见原因：DNS 不指向本机、云厂商安全组没放 80、Let's Encrypt 限流（等 1 小时） |
| `container did not reach 'healthy'` | 嵌入 key 无效 / 网络不通 / 磁盘满 | `docker compose --env-file ../.env logs --tail 80 aml-memory`，检查 `EMBED_API_KEY` 与出网 |
| 构建卡在 `pip install` 或超时 | 无法访问镜像源 | 重跑（脚本默认清华源）；仍失败则 `docker compose --env-file ../.env build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` |
| 自检某项 FAIL，例如 `add did not echo request_id` | 镜像版本不对 | `git -C /opt/aml-memory describe --tags` 必须是 `v1.2`；否则 `git fetch --tags && git checkout v1.2` 后重跑 |
| 自检 `search latency p50` 超 2000 ms | 嵌入/重排接口慢或机器过载 | 看 `docker stats`；必要时在 `.env` 里把 `EMBED_CONCURRENCY` 降到 4 |
| 容器被杀（OOM） | 内存 < 2 GB 或用户数据过大 | 加内存，或在 `.env` 里设 `AML_MEMORY_LIMIT=3g`（4 GB 机器）、`AML_INDEX_CACHE_VECTORS=60000` |
| `docker: command not found` 且安装失败 | 出网受限 | 让人类用云厂商镜像装 Docker，或改用阿里云 ECS 自带的 docker-ce |

任何一步失败，把**完整命令 + 完整输出**记录下来再回报，不要凭猜测"再跑一次试试"。

---

## 8. 完成报告模板（照抄回填给人类）

```
状态：部署完成 / 卡在第 X 步
机器：<云厂商/规格/公网IP>
域名：<AML_DOMAIN>（解析 IP：<ip>，与实际是否一致：是）
版本：aml-memory v1.2（commit <short>）
容器：aml-memory <healthy|...>，aml-caddy <running|...>
自检：13/13 PASS（或：第 N 项 FAIL，原文：...）
HTTPS：https://<域名>/health -> 200（或错误码）
榜单要填的值：
  Add    https://<域名>/add
  Search https://<域名>/search
  Health https://<域名>/health
  Token  <token>
  Track  Textual Memory / Division: Open-source Methods
  Embedding text-embedding-v4 (1024-d) / LLM: N/A / Reranker: gte-rerank-v2
待人类处理：榜单注册、Smoke、Full
```

---

## 附：日常运维

```bash
cd /opt/aml-memory/deploy
docker compose --env-file ../.env ps                       # 状态
docker compose --env-file ../.env logs -f --tail 100 aml-memory
curl -s http://127.0.0.1:8000/admin/stats -H "authorization: Bearer $TOKEN"   # units/users/vectors 计数
docker compose --env-file ../.env restart aml-memory       # 重启（数据在卷里，不丢）
docker run --rm -v deploy_aml-data:/data -v "$PWD":/backup alpine \
  tar czf /backup/aml-data-$(date +%F).tgz -C /data .      # 备份（建议每天一次）
git -C /opt/aml-memory describe --tags                     # 当前版本，必须始终是 v1.2
```

评测期间（人类跑 Full 时）**不要**升级版本、不要改 `.env`、不要重启机器：平台在评测窗口内认定版本冻结。

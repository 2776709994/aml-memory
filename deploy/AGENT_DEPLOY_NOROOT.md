# 无 root / 无 Docker 部署作业单（Cloudflare Tunnel 版）

你在一台**只有普通账号、没有 sudo、没有 Docker** 的 Linux 服务器上，要把 aml-memory
跑起来，并让它有一个**公网 HTTPS 地址**。全部工作由 `deploy/bootstrap-noroot.sh` 完成，
本文告诉你需要什么、怎么判断成功、什么时候必须停下。

> 适用条件：不能装 Docker、不能绑 80/443、不能改系统。缺点是服务靠普通进程 + 看护循环跑，
> 机器重启后需要手动或靠 crontab 拉起来（附录 A），所以**这台机器在评测期间要一直开着**。

---

## 0. 一句话目标

用普通账号把服务跑在 `127.0.0.1:8000`，用 cloudflared 命名隧道把它发布成
`https://<你的子域>/add`、`/search`、`/health`，通过 15 项契约自检，然后**停下**等人类做榜单注册。

---

## 1. 开工前：人类必须先做完这四件事（Agent 做不了）

| # | 事项 | 交给你的东西 |
|---|---|---|
| 1 | 域名托管到 Cloudflare（NS 已生效，`dig NS 你的域名` 能看到 `*.ns.cloudflare.com`） | 子域名，例如 `memory.example.com` |
| 2 | Cloudflare Zero Trust → **Networks → Tunnels → Create a tunnel**（类型 `cloudflared`），创建后复制安装命令里的 **token**（`eyJ...` 长串） | `CF_TUNNEL_TOKEN` |
| 3 | 在该隧道里加 **Public Hostname**：子域 `memory` + 你的域名 → Service 选 `HTTP` → URL 填 `localhost:8000` | 配好即可，无需给你 |
| 4 | 能调用 `text-embedding-v4` 的 DashScope Key（`sk-` 开头）；可选：托管重排 `gte-rerank-v2` 用同一个 Key | `EMBED_API_KEY`（`RERANK_API_KEY` 可选） |

> **密钥必须是长期的。** 这个服务唯一的强外部依赖就是嵌入接口：key 在 Full 评测跑到一半失效，
> 那一次机会就作废了（每个 Key 只允许 2 次 Full）。交付前让人类确认 key 来自按量付费的
> Model Studio 正式 API-KEY 页面，而不是临时/工作空间临时凭证。脚本开跑时会先真打一次
> `text-embedding-v4`，401/403 立即退出并提示换 key。

顺序很重要：**先建隧道和 Public Hostname，再跑脚本**。反过来也能跑起来，但脚本最后一步的
公网健康检查会失败（隧道当时还不存在），你会以为部署失败了。

`localhost:8000` 是服务在本机监听的地址（脚本默认端口 8000，可用 `AML_PORT` 改；改了这里也要跟着改）。

如果你拿不到上面任何一项，**不要用占位符硬跑**，直接回报缺什么。

---

## 2. 环境自检（1 分钟，只读，不改任何东西）

```bash
whoami; id -u                      # 不需要 root，普通账号即可
nproc; free -m | head -2; df -h "$HOME" | tail -1
python3 -V                          # 需要 3.9+；3.10–3.12 最稳
python3 -m venv --help >/dev/null 2>&1 && echo "venv ok" || echo "venv 缺失（脚本有兜底，见 §7）"
command -v curl tar git openssl
ss -ltn | grep -E ':(8000)\b' || echo "8000 空闲"
```

判断标准：

* 内存 ≥ 2 GB、`$HOME` 剩余 ≥ 3 GB（**磁盘**：约 500 MB 依赖 + 每 100 万轮记忆约 1 GB 数据库）；
* `python3` 存在；没有 `venv` 模块也能继续（脚本会退到 `--without-pip` + get-pip，再退到用户级安装）；
* 8000 没有被别的进程占用（占用就换 `AML_PORT`）；
* 不需要 Docker，也不需要能访问 Docker Hub。

---

## 3. 执行部署（一条命令，可重复执行）

```bash
# 1) 取脚本（国内可用 ghfast 反代；海外可直接用 raw）
curl -fsSL https://ghfast.top/https://raw.githubusercontent.com/2776709994/aml-memory/v1.3/deploy/bootstrap-noroot.sh \
  -o /tmp/aml-bootstrap-noroot.sh
wc -c /tmp/aml-bootstrap-noroot.sh        # 应为 23084

# 2) 跑它（下面三个变量换成真实值）
AML_DOMAIN=memory.example.com \
CF_TUNNEL_TOKEN=eyJ... \
EMBED_API_KEY=sk-... \
bash /tmp/aml-bootstrap-noroot.sh
```

脚本按顺序做这些事（每一步都自带检查）：

1. 预检：python3、curl/tar、内存、磁盘、端口、域名与 token 的字符合法性，**并且真的用 `EMBED_API_KEY` 打一次 `text-embedding-v4`、用同一个 key 打一次 `gte-rerank-v2`**（嵌入 401/403 立刻报错退出，不会让你白等十分钟；重排失败只警告，检索会自动退回融合排序）；
2. 取源码到 `$HOME/aml-memory`（v1.3 标签；已有则原地更新）；
3. 建 Python 环境：优先用 3.10–3.12 建 venv，没有 ensurepip 就 `--without-pip` + get-pip，再不行退到用户级安装；
4. 装依赖（依次试清华/阿里/中科大/PyPI 四个源，一个卡住或失败自动换下一个，并打印最终生效的源）；
5. 生成 `$HOME/aml-memory/.env`：随机 `AML_API_KEY`（`chmod 600`）、嵌入 `text-embedding-v4`、重排 `gte-rerank-v2`（**默认复用同一个 key**；想关掉就传 `RERANK_API_KEY=` 并手工把 `RERANK_PROVIDER` 改成 `none`）；**文件已存在时是"合并"而不是覆盖**：已有 token 和已填的值一律保留，只补空的和缺的键（早期用临时隧道跑过一次、留下 `AML_DOMAIN=` 和 `CF_TUNNEL_TOKEN=` 空值的机器，重跑一次就会自动补齐）；
6. 下载 `cloudflared` 到 `$HOME/aml-memory/bin/`（ghfast 反代 + 官方地址两路重试）；
7. 启动服务和隧道（都由看护循环托管，挂了 3–5 秒自动重起）；
8. 等本机 `/health` 200 → 等 `https://<域名>/health` 200；
9. 在服务器上直接跑 15 项契约自检；
10. 打印要填进榜单的 URL 与 token。

预计耗时：首次 3–8 分钟（依赖下载占大头）；重复执行 1–2 分钟。

**已经部署过、只想升级到新代码**：`AML_DOMAIN` / `CF_TUNNEL_TOKEN` / `EMBED_API_KEY` 会从已有的
`$AML_DIR/.env` 里自动读取，所以只要

```bash
bash /tmp/aml-bootstrap-noroot.sh
```

即可（`.env` 是合并、数据目录不动、服务自动重启；升级前建议先备份 `data/`，见附录 A）。

可选参数：`AML_PORT`（默认 8000）、`AML_TAG`（默认 v1.3）、`AML_DIR`（默认 `$HOME/aml-memory`）、
`AML_SRC`（离线从本地目录部署）、`PIP_INDEX_URL`、`PIP_SPEC`（默认锁定的 `requirements.txt`）、
`CF_QUICK=1`（**临时**隧道，见 §6）、`SKIP_VERIFY=1`、`CF_BIN`（已有 cloudflared 时直接指定）。

---

## 4. 判断成功（必须看到这三样）

1. 最后一行是 `Deployment complete.`，并打印出 Add/Search/Health 三个 URL 和 token；
2. 上面有 `all 15 checks passed - the endpoint is ready to register.`；
3. 下面三条命令的结果与期望一致：

```bash
cd "$HOME/aml-memory"
deploy/service-noroot.sh status                    # api healthy / tunnel running / public https://...
curl -s -o /dev/null -w '%{http_code}\n' https://memory.example.com/health          # 200
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://memory.example.com/search \
     -H 'content-type: application/json' -d '{"query":"x","user_id":"y"}'           # 401
```

随时可以重跑自检（不破坏数据）：

```bash
TOKEN=$(grep '^AML_API_KEY=' "$HOME/aml-memory/.env" | cut -d= -f2-)
"$HOME/aml-memory/.venv/bin/python" "$HOME/aml-memory/scripts/verify_endpoint.py" \
  --base-url https://memory.example.com --token "$TOKEN"
```

---

## 5. 停止线（到这里必须停，交回人类）

1. **不要**去 AML 平台注册、填 URL/token、申请 Key；
2. **不要**自己发起 Smoke 或 Full 评测（Full 每个 Key 只有 2 次机会，第二次要等 30 天，且必须跑到 100%）；
3. **不要**为了让自检通过去改服务代码或调检索参数；
4. **不要**重启机器、升级系统、换 Python 版本（评测窗口内版本冻结）。

---

## 6. 禁止事项

* 不要尝试 `sudo`、不要改系统包、不要绑 80/443（本方案不需要）；
* 不要把 8000 端口暴露到公网（只让 cloudflared 反代；防火墙保持关闭入站）；
* 不要设 `AML_DISABLE_AUTH=1`，不要删除或覆盖 `.env` 里的 `AML_API_KEY`；
* 不要删除 `$HOME/aml-memory/data/`（全部记忆都在 `aml.sqlite3` 里；要备份见附录 A）；
* 不要在 Full 评测期间用 `CF_QUICK=1`：临时隧道每次重启都换域名，域名一换榜单里填的 URL 就失效；
* 不要 `git push`、不要把 `.env` 提交进任何仓库；
* 不要在评测期间 `docker`/`pip`/系统大升级，也不要让机器休眠或关机。

---

## 7. 故障对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `cannot create a Python environment` | 没有 venv 模块、没有 ensurepip、也没有 pip | 让管理员装 `python3-venv`；或在一台能建 venv 的机器上准备好 `AML_SRC` 目录（含 `.venv`）后重跑 |
| `pip install failed` | 出网受限/索引不通 | 重跑；或 `PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/`；Python ≥3.13 报没有轮子时用 `PIP_SPEC="fastapi uvicorn httpx numpy pydantic python-dotenv orjson"`（不锁版本） |
| `could not download cloudflared` | GitHub 不可达 | 手动下载 `cloudflared-linux-amd64` 到服务器，`CF_BIN=/path/to/cloudflared` 后重跑 |
| `public health check failed for https://…` | 隧道或 Public Hostname 没配好；域名 NS 没生效 | `tail -f $HOME/aml-memory/logs/tunnel.log`；确认 Public Hostname 指向 `http://localhost:8000`；`dig +short 你的域名` |
| 浏览器访问返回 **502** | cloudflared 在跑但服务没起 | `deploy/service-noroot.sh status`；`tail -50 logs/api.log`（常见：`EMBED_API_KEY` 无效导致启动即退出） |
| 本地 `/health` 200，公网也 **502** | Public Hostname 的 Service Type 选成了 `HTTPS`，而服务是纯 HTTP | Cloudflare → 该隧道 → Public Hostname 里把 **Service Type 改成 `HTTP`**、URL 保持 `localhost:8000`，保存后立即生效（实测踩过） |
| `no healthy answer on 127.0.0.1:8000` | 服务启动失败 | 同上；注意端口是否被占用（换 `AML_PORT`） |
| 自检某项 FAIL | 版本不对或环境被改 | `git -C $HOME/aml-memory describe --tags` 必须是 `v1.3`；否则 `git fetch --tags && git checkout v1.3` 后重跑 |
| `the embedding key was rejected (HTTP 401)` | 密钥失效/填错/已过期 | 到 Model Studio 控制台重新签发一个**长期** key（不要用临时 workspace key），`EMBED_API_KEY=sk-... bash /tmp/aml-bootstrap-noroot.sh` 重跑即可（`.env` 已存在时会保留 token，只改嵌入部分需手工 `sed` 或删 `.env` 重生成——注意删 `.env` 会换新 `AML_API_KEY`，榜单里也要同步改） |
| `/add` 返回 500，日志里是 `embedding HTTP 401` | 部署完成之后 key 才失效（本方案唯一的强外部依赖） | 换新 key：`sed -i 's|^EMBED_API_KEY=.*|EMBED_API_KEY=sk-新key|' ~/aml-memory/.env && deploy/service-noroot.sh restart`；**Full 开始前务必确认 key 能长期使用**，跑到一半失效等于浪费一次机会 |
| 服务被系统杀掉（OOM） | 内存不足 | `.env` 里设 `AML_INDEX_CACHE_VECTORS=30000`、`AML_CANDIDATE_DEPTH=300` 后 `deploy/service-noroot.sh restart`；或换更大内存的机器 |
| 机器重启后服务没了 | 没有 systemd 单元 | 见附录 A：有 cron 就 `@reboot`，没有就让外层巡检每几分钟调一次 `deploy/watchdog.sh`（它只在真挂时动手） |
| `.env` 里 `AML_DOMAIN=` / `CF_TUNNEL_TOKEN=` 是空的 | 早期用 `CF_QUICK=1` 跑过一次，旧版本脚本遇到已存在的 `.env` 会整份跳过 | 直接带着正确的 `AML_DOMAIN` 和 `CF_TUNNEL_TOKEN` 重跑脚本即可（新版是合并，只补空值、token 不变）；或手工 `sed` 填这两行后 `deploy/service-noroot.sh restart` |
| `port 8000 already in use` | 端口冲突 | `AML_PORT=8010 …` 重跑，并同步改 Cloudflare 的 Public Hostname |

---

## 8. 回报模板（部署完照这个格式发回来）

```
状态：成功 / 失败（第 N 步）
版本：git -C ~/aml-memory describe --tags
URL ：Add / Search / Health 三条
自检：all 15 checks passed（或：第 N 项 FAIL，原文：...）
进程：api healthy / tunnel running；pid ...
机器：python3 -V；内存 X MB；磁盘剩余 Y GB
公网：curl -s -o /dev/null -w '%{http_code}' https://<域名>/health → 200
注意：<任何异常、你绕过的地方、剩余的担心>
```

---

## 附录 A：运维命令

```bash
cd "$HOME/aml-memory"

deploy/service-noroot.sh status      # 服务/隧道/公网地址一眼看全
deploy/service-noroot.sh logs        # 跟踪 API 日志（Ctrl-C 退出，不影响服务）
deploy/service-noroot.sh restart     # 重启（数据在 data/aml.sqlite3，不丢）
deploy/service-noroot.sh stop        # 停
deploy/service-noroot.sh url         # 打印当前公网地址
tail -f logs/tunnel.log              # 隧道日志

# 备份（数据库一致性靠 SQLite 的 WAL，先停 1 秒最稳）
deploy/service-noroot.sh stop && tar czf "$HOME/aml-backup-$(date +%F).tgz" -C "$HOME/aml-memory" data .env \
  && deploy/service-noroot.sh start

# 开机自启（没有 systemd 权限时）
( crontab -l 2>/dev/null; echo "@reboot cd $HOME/aml-memory && START_TUNNEL=1 AML_PORT=8000 deploy/service-noroot.sh start" ) | crontab -

# 没有 cron 也没有 systemd 时：让外层看护（你自己的巡检、heartbeat 之类）每次调这个
deploy/watchdog.sh                   # 健康就不动手；API 挂了自动 restart，隧道掉了自动拉起；退出码 0=正常 1=仍然坏

# 换（轮换）嵌入/重排 key：显式传的值会覆盖 .env，其余配置与数据都不动
EMBED_API_KEY=sk-新key bash /tmp/aml-bootstrap-noroot.sh
# 若新 key 不是这个工作区（MaaS）的，还要同时指到官方地址：
# EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 EMBED_API_KEY=sk-新key bash /tmp/aml-bootstrap-noroot.sh
```

> 轮换 key 时脚本会先拿新 key 真打一次 `text-embedding-v4`（和一次 `gte-rerank-v2`），401 就当场停下，
> 不会把线上打成 500；成功后自动重启服务，`AML_API_KEY`（平台用的 token）、域名、隧道 token 全部保留。

## 附录 B：为什么这台机器要一直在线

平台按自己的排期调用你的端点：Smoke 几分钟就结束，**Full 是对约 6,000 个实例、约 3 亿 token 的批量评测**，
可能持续数小时到数天。平台对超时/5xx 会带同一个 `request_id` 重试（最多 32 次），所以**几秒到几分钟的抖动没事**；
但如果机器休眠、断电、断网几小时，这个阶段就会作废，而 Full 每个 Key 只有 2 次机会。

因此：插电、关掉休眠（`sudo` 不行时至少别手动合盖/关机）、别在评测期间升级系统。

## 附录 C：换到别的机器（或 VPS）

同一套脚本，换机器重跑即可：

```bash
curl -fsSL https://ghfast.top/https://raw.githubusercontent.com/2776709994/aml-memory/v1.3/deploy/bootstrap-noroot.sh -o /tmp/n.sh
AML_DOMAIN=memory.example.com CF_TUNNEL_TOKEN=eyJ... EMBED_API_KEY=sk-... bash /tmp/n.sh
```

Cloudflare 侧的 Public Hostname 不用改（隧道 token 一样，connector 换机器就是换出口）。

如果有 root 和 Docker，改用 `deploy/bootstrap.sh`（Caddy 自动 HTTPS、不需要 Cloudflare 账号）。

# 发布基础设施草案（尚未启用）

目标仅为 `WaydeWan/AgenthonT2---ReasoningAugmentedTimeSeries---Money-Miner` 的三套最终候选镜像。当前候选尚未冻结，模板 `candidates: []` 会拒绝构建。没有上传、登录、发布、修改 package visibility 或读取用户凭据。

## 最终应用顺序

1. 固定三种实际配置、CLI 入口、所有拟合 artifact 和许可说明。填写 `release-manifest.json`；不得把研究历史、原始训练行、数据缓存、证据目录或完整 private 仓库放进名单。
2. 主代理完成本地正式检查，再运行 `release.py export` 到一个**不存在的新目录**。导出仅复制名单中的文件、锁定的官方运行源码和本草案明确列出的基础设施；不会复制 Git 历史，也不会初始化 Git 或 push。审阅新目录后，才由主流程将这些文件用于已授权的公开仓库。
3. 默认分支的手动 Actions workflow 会再次校验三配置和所有 SHA，分次创建三个独立构建上下文，推送 `linux/amd64`。版本 tag 使用 `release_id + candidate id + config hash`，最终提交必须使用构建输出的 digest。三个 `published-*` artifact 各含一个 `build-record.json`；保留三份，不把局部成功写成全部成功。
4. 首次 push 后确认 GHCR package 的公开可见性。仓库公开不代表镜像公开。再对**每份实际构建输出**执行匿名 pull 验证，保存新证据文件。失败退出码为 2，不能充作提交证据。
5. 用相同 digest 跑正式接口与官方 g0–g3 smoke；本草案只验证镜像拉取、平台和接口标签，不执行模型，也不替代 smoke、模型披露、cutoff 审计、评分或 ZIP 封装。

草案并不把某个旧模型名单当作最终三种。配置相同而仅换名称会被拒绝；最终三配置是否有实质预测差异仍需研究与发布审查。

## Manifest 合同

顶层字段必须恰为 `schema_version`、`repository`、`release_id`、`files`、`candidates`，见空模板。`release_id` 和三个不同的 `id` 必须为小写字母开头、最多 40 位的字母/数字/连字符。

- `files`：逐文件 `{ "path": "项目相对路径", "sha256": "真实文件 SHA256" }`，不支持 glob。团队源只允许 `submission_candidates/` 下非测试 `.py`、`release_configs/` 与 `release_artifacts/` 下明确审核的 `.json`，以及一个团队 `LICENSE`（根路径或旧 bundle 的确切路径）。全部代码和 artifact 都会公开，请人工审阅 JSON 不含训练历史；路径过滤与 token 模式扫描并非任意敏感内容的证明。
- 每个 candidate：仅 `{ "id": "最终配置标识", "config": "submission_candidates/release_configs/具体配置.json" }`。该配置必须已列入 `files`。
- 配置 JSON：仅 `module`（真实 `submission_candidates.*` 模块）与 `environment`。环境只允许 `MONEY_MINER_*` 非凭据/非远程 API 地址字段；值是短字符串。使用 `/app/...` artifact 路径时，该文件必须列入白名单。不要将 House 凭据固化进镜像；官方运行环境负责注入。
- 官方代码另由 `official-sources.lock.json` 列出的 127 个源/元数据/许可文件逐字节锁定：toolkit `2.6.0`，Track 2 commit `60509df4ad0756443f4af8dc9500aed99a995694`。不复制其数据、题库或测试。锁文件来自已审阅的本地官方快照；不是一次新的线上全仓库真实性证明。

正式命令入口（从项目根目录运行；路径按最终 manifest 替换）：

```text
python -m unittest discover -s delivery/publish_scaffold_v1/tests -v
python delivery/publish_scaffold_v1/release.py matrix --manifest release-manifest.json
python delivery/publish_scaffold_v1/release.py export --manifest release-manifest.json --out delivery/public-release-export
```

匿名拉取命令中的 `ACTUAL_DIGEST` 必须取自真实 build record；这里没有示例假 digest：

```text
python delivery/publish_scaffold_v1/verify_anonymous_pull.py --image ghcr.io/waydewan/agenthont2---reasoningaugmentedtimeseries---money-miner@sha256:ACTUAL_DIGEST --out evidence/anonymous-pull-CANDIDATE.json
```

匿名工具创建新的空 `config.json`，不继承 credential helper、Docker context、环境 token 或旧登录，调用本机默认 Docker daemon。Linux 镜像须由可用的 Linux Docker daemon 拉取。不要用携带认证上游镜像缓存的 daemon；工具证明客户端未提供凭据，不能审计不受控制的服务端代理。即使本机已有层，`docker pull` 仍必须成功后才检查本地平台、RepoDigests 和标签。

## GitHub 权限的实际边界

Workflow 只有手动入口和默认分支，checkout 不持久化凭据；发布 job 仅 `contents: read` 与 `packages: write`，使用 GitHub 自带短期 `GITHUB_TOKEN`，不需要把个人 PAT、团队 claim key、NVIDIA key 放进仓库或构建参数。发布镜像含 `org.opencontainers.image.source` 关联目标仓库。[GitHub 的容器发布方式](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)。

个人 GHCR package 首次默认 private；可由 package admin 在 **Package settings → Change visibility → Public** 设置公开。已有同名 package 若未继承本仓库权限，则在 **Manage Actions access** 给该仓库 write。管理员可能需要允许本 workflow 用到的官方 Actions 和 `packages: write`；草案未查验账户设置或改变它们。公开 package 才支持匿名 pull。[GitHub 的 package 权限与可见性说明](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility)。这些是发布时可能需要账户所有者操作的点，现在不预先请求权限。

## 复现和公开边界

所有外部 Actions 均固定到官方仓库的完整 commit SHA，版本和来源见 `audit.json`。Python 基础镜像沿用现有 base 的 digest；直接数值依赖版本沿用现有 base。pip 的传递依赖、构建后端、GitHub runner 与 BuildKit 并非全部逐制品 hash 锁定，因此不宣称多次构建字节相同；实际 digest 才是提交身份。Buildx 固定 `v0.23.0`，native amd64 无需 QEMU。

团队源码、配置、JSON 拟合参数、官方源码与许可证均对下载者可见；不上传 build context artifact、源码全量 artifact 或 Docker 构建日志制品。GitHub 正常运行日志仍公开。发布记录没有模型预测准确性结论；匿名验证通过也不是参赛资格认证。最终私有审计证据和团队 claim 密钥留在本地。

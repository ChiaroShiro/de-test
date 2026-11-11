# Mac 配置 Codex 和 DSH 中转：API Key 写入文件的逐步教程

核对日期：2026-11-11。适用于已经安装 Node.js、Codex CLI 和 DeepSeek Harness CLI 的 Mac，终端使用默认的 zsh。

这份教程沿用当前 Windows 本机的中转地址、协议和默认模型，在 Mac 上填写各自的新 API Key。教程中的 `YOUR_CODEX_KEY` 和 `YOUR_DSH_KEY` 都是占位符，必须替换；本文没有包含本机真实 Key。

## 1. 先弄清楚要改哪几个文件

Key 可以保存在文件里。本机目前就是这样做的：Codex 的 Key 直接放在 `config.toml`，DSH headless 的 Key 放在 `.dsh/.env`。

此前写入 `~/.zshrc` 的方案也会把 Key 保存到磁盘，但需要由终端把它导出成环境变量。本教程改为让工具自己读取配置文件，不需要每次执行 `export` 或 `source`。

| Mac 上的文件 | 要放什么 | 需要你替换的内容 |
|---|---|---|
| `~/.codex/config.toml` | Codex 的地址、模型、协议、Key | `YOUR_CODEX_KEY` |
| `~/.dsh/.env` | DSH 的 Key | `YOUR_DSH_KEY` |
| `~/.dsh/cordis.patch.yml` | DSH 的地址、模型、协议和 Key 的字段名 | 按示例填写，无须填真实 Key |
| `~/.zshrc` | 清理此前增加的两条 Key 导出语句 | 只删除对应两行，保留其他配置 |

`~` 代表你的 Mac 用户主目录，例如 `/Users/tyh`。教程中的命令直接输入 `~` 即可，不必换成用户名。点号开头的文件在访达中默认隐藏，终端仍能打开。

当前核对出的连接参数如下：

| 项目 | Codex | DSH |
|---|---|---|
| API 基础地址 | `https://cpa0.shgao.top/v1` | `http://112.126.71.3:25300/v1` |
| 协议 | `responses` | `openai-completions` |
| 默认模型 | `gpt-6-astra` | `deepseek-v4.1-flash` |
| 推理等级 | `xhigh` | 沿用本机中转配置，不额外设置 |

本机 DSH 的上述中转配置用于 `headless`；本机 `web` 当前选择的是 DeepSeek 账号登录。本教程把中转写入 Mac 的 DSH 全局配置，因此 Mac 的 `web` 和 `headless` 都能使用它。

## 2. 操作前的准备

### 2.1 先退出正在运行的工具

如果终端正在运行 Codex 或 `dsh web`，先按 `Control + C` 退出；若仍未回到普通命令提示符，再按一次。浏览器中的 DSH 网页先关闭。

以下标为 `bash` 的代码块是终端命令：每输入一行，按一次回车。标为 `toml`、`dotenv`、`yaml` 的代码块是要写进编辑器的文件内容，不能作为终端命令执行。不要输入 Markdown 的三个反引号。

### 2.2 确认命令可用

逐行输入：

```bash
node -v
npm -v
codex --version
dsh --version
```

四条命令应分别显示版本号。若 `codex` 或 `dsh` 尚未安装，分别安装：

```bash
npm i -g @openai/codex
npm i -g @deepseek-ai/dsh
```

若已安装，无须重复安装。遇到 `EACCES` 或 `command not found`，先看本文第 8 节对应条目。

### 2.3 创建配置目录

输入：

```bash
mkdir -p ~/.codex ~/.dsh
umask 077
```

第一行在目录不存在时创建它们；第二行让本次终端会话中新建的普通文件默认只对自己开放。

本教程使用默认目录。如果你曾主动设置 `CODEX_HOME` 或 `DSH_HOME`，实际配置位置可能不同；可输入以下两行检查。输出为空表示没有设置，可以直接继续；有输出时，应使用输出所指向的目录。

```bash
printenv CODEX_HOME
printenv DSH_HOME
```

### 2.4 认识 nano 的保存方法

本教程使用 Mac 自带的终端编辑器 `nano`，不需要额外安装编辑器。

每次编辑完都这样保存：

1. 按 `Control + O`，这里是字母 O，不是数字 0，也不是 Command。
2. 底部会出现文件名，直接按回车确认。
3. 按 `Control + X` 退出编辑器。

用方向键移动光标，普通退格键删除文字；`Control + K` 可以删除光标所在的一整行。若终端窗口较窄，长行可能只显示一部分，不要因为屏幕换行就自行插入回车。

## 3. 清理上一版写在 .zshrc 里的 Key

如果你没有执行过上一版的 Key 配置，可跳过本节。

先备份，再打开文件：

```bash
cp -n ~/.zshrc ~/.zshrc.before-file-key
nano ~/.zshrc
```

`cp -n` 在备份已存在时保留旧备份，不覆盖它。如果提示原文件不存在，说明该文件尚未创建，这一步可跳过。

找到之前增加的以下两类语句，实际引号中可能已经是你的 Key：

```bash
export CODEX_RELAY_KEY='之前填写的Codex中转Key'
export DEEPSEEK_API_KEY='之前填写的DSH中转Key'
```

只删除这两行，其他内容保留。尤其保留之前为解决 npm 路径问题添加的这一行：

```bash
export PATH=~/.local/bin:$PATH
```

保存退出后，在终端逐行输入：

```bash
unset CODEX_RELAY_KEY
unset DEEPSEEK_API_KEY
```

这会清除当前终端里已经加载的旧值。只删 `.zshrc` 中的文字，并不会立即清掉当前终端的环境变量。其他已经打开的终端窗口应关闭后重新打开。

## 4. 先配置 Codex：Key 直接写在 config.toml

### 4.1 备份已有配置

如果这个文件已存在，输入：

```bash
cp -n ~/.codex/config.toml ~/.codex/config.toml.before-file-key
```

如果提示 `No such file or directory`，说明还没有配置文件，直接进入下一步。

### 4.2 打开配置文件

```bash
nano ~/.codex/config.toml
```

### 4.3 写入下面的配置

如果文件为空，直接输入完整内容。如果文件只有之前教程中的最小配置，可将它调整成下面的完整内容。若已经有其他 MCP、项目或个人设置，保留那些配置，按照第 4.4 节修改连接部分。

```toml
model = "gpt-6-astra"
model_provider = "relay"
model_reasoning_effort = "xhigh"

[model_providers.relay]
name = "relay"
base_url = "https://cpa0.shgao.top/v1"
wire_api = "responses"
requires_openai_auth = false
experimental_bearer_token = "YOUR_CODEX_KEY"
```

**把最后一行的 `YOUR_CODEX_KEY` 替换成 Mac 要用的 Codex 中转 Key，保留左右的英文双引号。** 不要额外加 `Bearer ` 前缀，不要填写 DSH 的 Key。

例如，最后一行的形式应该是 `experimental_bearer_token = "你自己的完整Key"`。这只是格式说明，中文文字也必须替换成真实 Key。

这里的 `relay` 是本地提供商名称，可以与 Windows 上的 `OpenAI` 名称不同；实际连接由地址、协议和 Key 决定。文件中的 `model_provider = "relay"` 与 `[model_providers.relay]` 必须对应。

`experimental_bearer_token` 是 Codex 支持的直接保存令牌字段。官方更推荐 `env_key`；本教程按你的要求使用直接写文件的方式，与当前 Windows 本机的认证方式一致。[Codex 官方配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)

### 4.4 如果已有配置，具体改哪里

| 原内容或情况 | 操作后的内容 |
|---|---|
| `model` 是其他值 | 修改为 `model = "gpt-6-astra"` |
| `model_provider` 是其他值 | 修改为 `model_provider = "relay"` |
| 缺少推理等级，或值不同 | 添加或修改为 `model_reasoning_effort = "xhigh"` |
| `[model_providers.relay]` 中有 `env_key = "CODEX_RELAY_KEY"` | 删除该行，换成 `experimental_bearer_token = "YOUR_CODEX_KEY"`，并填入真实 Key |
| 同一个 relay 段内有其他 `env_key` | 删除这条 `env_key`，使用上面的直接令牌字段 |
| `requires_openai_auth = true` | 修改为 `requires_openai_auth = false` |
| 没有 `[model_providers.relay]` | 在文件末尾增加第 4.3 节中的整个 relay 段 |
| 已有 `[model_providers.relay]` | 修改原来的段，不要再次添加同名段 |
| 存在 `[model_providers.relay.auth]` 及其命令认证字段 | 在备份后移除这一段，避免与直接令牌认证混用；不要删除其他提供商的 auth 段 |

前三个 `model...` 设置必须处在文件顶层，也就是第一个 `[某个段名]` 之前。不能把它们写进 `[model_providers.relay]` 或 `[mcp_servers...]` 下面。对于缺失的顶层字段，将它们加入文件顶部；已有字段直接修改，不能重复添加。

`[model_providers.relay.auth]` 这样的段从段名开始，到下一个 `[段名]` 之前结束；没有下一个段名时，到文件末尾结束。

本方案直接从 `config.toml` 取 Key，不需要另外填写 `auth.json`，也不需要把中转 Key 填到 ChatGPT 登录网页。Mac 以前已有的 `auth.json` 可以保留。

### 4.5 保存并设置文件权限

按 `Control + O`、回车、`Control + X`，然后输入：

```bash
chmod 600 ~/.codex/config.toml
```

这让该配置文件只允许当前用户读写。

### 4.6 启动验证

```bash
codex
```

新建会话，输入 `/status` 查看当前模型是否为 `gpt-6-astra`；如果界面显示 provider，应为 `relay`。然后发送：

```text
只回复“你好”，不要调用工具。
```

收到正常回复才说明模型请求已经成功。单纯能打开 Codex 界面不代表 Key 有效。该测试会发送一次实际模型请求。

## 5. 再配置 DSH：Key 写在 .env，地址写在 YAML

DSH 的连接配置使用 `apiKeyEnv` 指定凭据名称，它不是填写真实 Key 的位置。真实 Key 放在 DSH 会自动读取的 `~/.dsh/.env` 文件里；本教程不需要把这个文件 `source` 到终端。

默认 headless 会读取启动环境、当前工作目录的 `.env` 和 DSH 主目录的 `.env`，不会使用网页保存的登录凭据。这也是这里选用 `.env` 的原因。[DSH 官方 headless 凭据说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/bundle/headless/README.md#credentials)

### 5.1 备份已有 DSH 配置

对已经存在的文件分别执行：

```bash
cp -n ~/.dsh/.env ~/.dsh/.env.before-file-key
cp -n ~/.dsh/cordis.patch.yml ~/.dsh/cordis.patch.yml.before-file-key
```

提示 `No such file or directory` 的那一个文件还没有创建，跳过对应备份即可。

### 5.2 打开 Key 文件

```bash
nano ~/.dsh/.env
```

如果文件为空，输入这一行：

```dotenv
DEEPSEEK_API_KEY=YOUR_DSH_KEY
```

**把 `YOUR_DSH_KEY` 替换为 Mac 要用的 DSH 中转 Key。** 直接放在等号后面，整条 Key 保持在同一行，不加 `export`，不加 `Bearer ` 前缀。

如果已有 `DEEPSEEK_API_KEY=...`，只替换等号后面的旧 Key；保留其他字段，不要重复增加另一条同名配置。

保存退出，然后输入：

```bash
chmod 600 ~/.dsh/.env
```

### 5.3 打开 DSH 的全局连接配置

```bash
nano ~/.dsh/cordis.patch.yml
```

注意路径是 `~/.dsh/cordis.patch.yml`，不是 `~/.dsh/profiles/web/cordis.patch.yml`。主目录的 patch 在各 profile 的 patch 之后加载，因此下面的设置会应用于使用该 DSH 主目录的 `web`、`headless` 等模式。[DSH 官方配置加载顺序](https://github.com/deepseek-ai/deepseek-harness/blob/master/apps/cli/README.md#profiles)

### 5.4 写入连接配置

如果文件为空，输入下面的完整内容。如果文件只有上一版教程中的配置，可以将其调整为下面的内容。缩进全部使用空格，每层两个空格，不使用 Tab。

```yaml
- id: llm-pi-ai
  config:
    providers:
      deepseek:
        apiKeyEnv: DEEPSEEK_API_KEY
        api: openai-completions
        baseURL: http://112.126.71.3:25300/v1
        models:
          - id: deepseek-v4-flash
          - id: deepseek-v4.1-flash
          - id: deepseek-v4-flash-fast

- id: agent-default-model
  config:
    provider: deepseek
    model: deepseek-v4.1-flash
```

三个模型 ID 来自本机现有配置，默认使用中间的 `deepseek-v4.1-flash`。模型是否都能调用取决于 Mac 上新 Key 的权限。

### 5.5 各个字段具体怎么填

| 字段 | 应填写的值 | 注意事项 |
|---|---|---|
| `providers` 下的提供商名称 | `deepseek` | 与默认模型的 `provider: deepseek` 对应 |
| `apiKeyEnv` | `DEEPSEEK_API_KEY` | 保留这个英文名称，不要替换成真实 Key；它对应 `.env` 中等号左边的字段 |
| `api` | `openai-completions` | 这是本机 DSH 中转实际使用的协议 |
| `baseURL` | `http://112.126.71.3:25300/v1` | `URL` 三个字母大写，地址末尾保留 `/v1` |
| 默认 `provider` | `deepseek` | 若以前是 `deepseek-account`，在此处改为 `deepseek` |
| 默认 `model` | `deepseek-v4.1-flash` | 若以前是 `deepseek-flash`，在此处改为这个完整 ID |

如果文件里已有 `- id: llm-pi-ai` 或 `- id: agent-default-model`，修改原条目，不要再添加同 ID 条目。如果 `llm-pi-ai` 下已经配置了其他提供商，保留它们，只添加或修改 `deepseek`。Cordis 覆盖可能替换完整条目的配置，不能用一个仅含 `deepseek` 的片段直接覆盖已有的多个提供商。[DSH 官方模型配置说明](https://deepseek-harness.github.io/deepseek-harness/guide/providers)

本教程没有额外设置 DSH 的 `reasoningEffort`。如果你在这个默认模型条目下从旧账号配置复制了 `reasoningEffort: high`，先移除这一行，以便先按本机中转配置验证。

保存退出。配置里的地址和模型不是 Key，真实 Key 只应写在第 5.2 节的 `.env` 文件中。

### 5.6 先验证 headless

先回到用户主目录，避免首次测试受到某个项目目录中 `.env` 的影响：

```bash
cd ~
unset DEEPSEEK_API_KEY
dsh --profile headless "只回复你好，不要调用工具"
```

`unset` 清除可能遗留的终端变量，不会删除 `.env` 中保存的 Key。这一步不需要重新 `export`。

正常情况下，DSH 回复后会自动退出，回到终端提示符。若主目录本身也有一个 `~/.env`，其中同名的 `DEEPSEEK_API_KEY` 仍可能覆盖 `~/.dsh/.env`，见第 8 节排查方法。

### 5.7 再验证网页模式

```bash
dsh web
```

1. 打开终端提示的网址；默认通常为 `http://127.0.0.1:3080`。
2. 新建会话，不要直接续用此前的账号登录会话。
3. 在模型选择器中选择提供商 `deepseek` 对应的 `deepseek-v4.1-flash`。显示名称可能被界面美化，核对模型 ID 和所配置的提供商。
4. 发送“只回复你好，不要调用工具”。
5. 保持运行 `dsh web` 的终端窗口打开；按 `Control + C` 会停止网页服务。

已有会话会保存自己的模型选择，因此修改默认配置之后，应通过新会话验证。不要选择 `deepseek-account` 对应的账号模型。[DSH 官方模型选择说明](https://deepseek-harness.github.io/deepseek-harness/guide/providers#选择模型)

## 6. 配置完成后，平时怎么用

配置文件会保留在 Mac 上。之后打开新终端，直接启动对应工具即可，不需要再次填写 Key，也不需要执行 `source ~/.dsh/.env`。

Codex：

```bash
codex
```

DSH 网页模式：

```bash
dsh web
```

DSH 单次命令行任务：

```bash
dsh --profile headless "你的任务描述"
```

在 Mac 上使用 `dsh`，不写 Windows 的 `dsh.cmd`。执行项目任务时，先进入你实际要操作的项目目录；第 5.6 节的 `cd ~` 只是首次连接测试使用。

## 7. 以后换 Key，只需要改哪里

### 更换 Codex Key

输入：

```bash
nano ~/.codex/config.toml
```

找到 `[model_providers.relay]` 中的：

```toml
experimental_bearer_token = "YOUR_CODEX_KEY"
```

把引号中的旧 Key 换成新 Key，保留英文双引号。保存后退出并重新启动 Codex。

### 更换 DSH Key

输入：

```bash
nano ~/.dsh/.env
```

找到 `DEEPSEEK_API_KEY=` 这一行，只替换等号后面的 Key。保存后停止并重新启动 `dsh web`；headless 下一次启动时会重新读取。无需修改 YAML 中的 `apiKeyEnv`。

## 8. 常见问题：看到什么，具体改什么

### 8.1 `npm`、`codex` 或 `dsh` 找不到

如果 `npm` 都找不到，重新打开终端再试；若仍找不到，从 Node.js 官网安装 macOS 的 Node.js 24.9 以上版本或更新的受支持 LTS 版本。[Node.js 官方下载页](https://nodejs.org/en/download)

如果 `npm` 可用、安装也已成功，但 `codex` 或 `dsh` 找不到，且你沿用了之前的 `~/.local` 安装目录，输入：

```bash
nano ~/.zshrc
```

确认里面有一条以下配置；没有时才添加：

```bash
export PATH=~/.local/bin:$PATH
```

保存后输入：

```bash
source ~/.zshrc
```

这里加载的是命令路径，不是 Key。

### 8.2 npm 安装报 `EACCES`，路径指向 `/usr/local`

输入：

```bash
npm config set prefix ~/.local
```

再按第 8.1 节配置命令路径，重新安装对应工具。该方法来自 [npm 官方权限错误说明](https://docs.npmjs.com/resolving-eacces-permissions-errors-when-installing-packages-globally/)。

### 8.3 Codex 报缺少 `CODEX_RELAY_KEY`

说明当前生效的提供商仍在读取环境变量。打开 `~/.codex/config.toml`，检查顶层是 `model_provider = "relay"`，并在 `[model_providers.relay]` 内把旧的：

```toml
env_key = "CODEX_RELAY_KEY"
```

替换为：

```toml
experimental_bearer_token = "YOUR_CODEX_KEY"
```

把占位符换成真实 Key。该 relay 段内不要同时保留 `env_key` 或命令式 `.auth` 配置。

### 8.4 Codex 仍要求登录，或模型不是配置的那个

核对 `model_provider` 在文件顶层，值为 `relay`；核对 `requires_openai_auth = false` 和直接令牌字段都在 `[model_providers.relay]` 内。完全退出后重新启动，不带旧的 `--profile` 或 `-c` 参数。若是首次启动的使用说明或项目授权提示，按提示完成即可；它与中转 Key 认证不是同一个步骤。

### 8.5 DSH 报 `MISSING_CREDENTIAL`

逐项检查：

1. 文件名是 `~/.dsh/.env`，不是 `.env.txt`。
2. 文件中有且只有一条有效的 `DEEPSEEK_API_KEY=完整Key`。
3. YAML 中写的是 `apiKeyEnv: DEEPSEEK_API_KEY`，拼写和大小写相同。
4. 执行一次 `unset DEEPSEEK_API_KEY`，再重启 DSH。
5. 若设置过 `DSH_HOME`，应在那个目录中的 `.env` 保存 Key。

不要用 `cat ~/.dsh/.env` 或打印环境变量的方式发送截图排错，这会显示完整 Key。

### 8.6 headless 正常，但网页仍使用旧 Key；或换目录后才报错

DSH 网页模式的凭据优先级是：已保存的凭据文件 → 启动环境 → 当前目录 `.env` → DSH 主目录 `.env`。因此，网页曾保存的旧 Key 或项目 `.env` 都可能覆盖这里的新 Key；默认 headless 不读取网页凭据文件。[DSH 官方凭据优先级](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/credentials/credentials-local/README.md#where-keys-come-from)

如果确实遇到覆盖，可给这条中转换一个独立的凭据名称，仍然把 Key 保存到文件：

1. 停止 DSH。
2. 打开 `~/.dsh/.env`，新增一行 `MAC_DSH_RELAY_KEY=你的DSH中转Key`；若已有这个字段，修改原行。
3. 打开 `~/.dsh/cordis.patch.yml`，只把 `apiKeyEnv: DEEPSEEK_API_KEY` 改成 `apiKeyEnv: MAC_DSH_RELAY_KEY`。
4. 地址、协议、提供商和模型都保持不变。
5. 在终端执行 `unset MAC_DSH_RELAY_KEY`，重新启动 DSH 并新建会话。

两处的 `MAC_DSH_RELAY_KEY` 必须完全相同，而且不要在其他启动配置、项目 `.env` 或网页凭据中重复设置这个新名称。使用此方法后，未来更换 DSH Key 应修改 `.env` 中的 `MAC_DSH_RELAY_KEY` 那一行。

### 8.7 报 `401`、`403`、`429`、模型不存在或连接超时

| 报错 | 优先检查 |
|---|---|
| `401` / `Unauthorized` | Key 是否输入完整、是否填到了对应工具、是否误留占位符或空格；也检查是否被旧凭据覆盖 |
| `403` / `Forbidden` | 新 Key 的模型、分组、IP 或账户权限是否允许这次请求 |
| `429` | 中转返回的说明是频率限制、并发限制还是额度问题 |
| `model not found` / `UNKNOWN_MODEL` | 模型 ID 拼写以及新 Key 是否开放 `gpt-6-astra` 或 `deepseek-v4.1-flash` |
| `404` | 基础地址是否精确包含 `/v1`，协议是否与本文一致；不要自行再拼 `/responses` 或 `/chat/completions` |
| 超时、无法连接、`Connection refused` | Mac 是否能访问中转服务、端口是否可达、服务是否运行 |
| `duplicate key` / TOML 解析错误 | 同名设置或同名 provider 段是否重复，是否误用了中文引号 |
| YAML 解析错误 | 是否用空格缩进、`- id:` 是否从行首开始、冒号后是否有空格 |

这些状态不能仅凭状态码确定唯一原因，应结合完整报错文字判断。新 Key 的权限和 Mac 的网络连通性只能在 Mac 上验证；本教程没有代替你发送认证请求。

## 9. 文件保存和适用范围

本教程使用默认用户目录，配置目标是终端启动的 Codex CLI 和 npm 安装的 DSH CLI；DSH 的独立桌面应用可能使用不同的 profile 或配置目录。

真实 Key 只填入 Mac 的配置文件，不要填回这份仓库教程。文件和备份都可能包含明文 Key，应留在个人配置目录。DSH 当前沿用的地址是 HTTP，传输没有 HTTPS 加密；只有中转方提供并确认 HTTPS 地址后，才能相应更换地址，不能简单把 `http` 改成 `https`。

本机配置已只读核对，文中的 TOML/YAML 示例可作为手动输入模板；Mac 上的实际版本兼容性、网络和新 Key 权限仍以第 4.6、5.6、5.7 节的请求结果为准。

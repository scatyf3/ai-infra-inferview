# AI Infra Inferview

北美 AI Infra 面试知识库。Markdown 文档 + 交互式可视化插件，部署在 GitHub Pages。

站点：<https://scatyf3.github.io/ai-infra-inferview/>

## 本地开发

```bash
npm install
npm run docs:dev      # http://localhost:5173/ai-infra-inferview/
npm run docs:dev:lan  # 同上，并监听局域网 / ZeroTier，手机可访问
npm run docs:build    # 生产构建（ignoreDeadLinks: false，坏链会失败）
npm run docs:preview  # 预览构建产物
npm test              # 计算逻辑的单元测试
```

## 结构

```
docs/                 VitePress 站点（srcDir）
  index.md            首页：推理栈分层图
  <domain>/*.md       7 个领域的知识文档
  .vitepress/
    domains.ts        领域元数据（唯一手工维护的结构信息）
    sidebar.ts        从 frontmatter 生成 sidebar
    data/topics.data.ts  构建时收集各页 status
    theme/components/    分层图 + 可视化插件
src/lib/              纯 TS 计算逻辑（可单测，不依赖 Vue/DOM）
src/data/             GPU 与模型规格表
templates/topic.md    写作模板
```

## 写一篇新文档

复制 `templates/topic.md` 到对应领域目录，填 frontmatter：

```yaml
---
title: 中文标题（术语保留英文）
status: todo        # todo | draft | reviewed —— 驱动分层图和进度面板里的状态点
tags: []
difficulty: 3       # 1–5，映射节点大小
order: 1            # 领域内排序
related: []         # 绝对路径，在地图上画虚线跨链
stack: [kv-paged, 5] # 挂到首页分层图：字符串 = 小主题 id，数字 = 整层综述；定义在 docs/.vitepress/layers.ts，空 = 不在图上
---
```

sidebar 和分层图都从 frontmatter 自动生成，不需要改配置。`status` 写错值会让构建失败。

分层图里每个小主题的基础介绍在 `docs/stack/<小主题 id>.md`，每层的总述在 `docs/stack/layer-<N>.md`：首页点格子或左侧层名时在本行下面展开，同时也是独立页面 `/stack/<id>`，底部自动列出 `stack` 挂到这里的文章。文件名必须对得上 layers.ts 里的小主题 id 或层号，否则构建失败。

手撕题（`docs/handson/`）多两项，驱动手撕首页的进度面板：

```yaml
familiarity: 2      # 熟练度 0 | 1 | 1.5 | 2 | 3 | 3.5 | 4（0 最熟），不写 = 未评；阶梯同 leetcode 看板
leetgpu: [50, 83]   # LeetGPU 题号，链接按 src/data/leetgpu-challenges.json 里的标题生成
```

两项写错值（不在阶梯上的熟练度、清单里没有的题号）同样会让构建失败。LeetGPU 出了新题时重新抓一次清单：
`curl -s https://api.leetgpu.com/api/v1/challenges`，按 id 排好只留 `id / title / difficulty / access` 覆盖 json。

## 可视化插件

全局注册，直接在 Markdown 里顶格写标签：

| 组件 | 用途 |
|---|---|
| `<MemoryCalculator />` | 显存账 + roofline：输入模型/GPU/batch/context，输出权重 KV 激活拆解、需要几张卡、prefill 与 decode 的 AI 落点 |
| `<ShapeFlow variant="gqa" phase="decode" />` | 一层 Transformer 的 tensor shape 流转，可切 MHA/GQA/MQA/MLA 与 prefill/decode |
| `<ParallelismViz :tp="4" :pp="2" />` | TP/PP/DP/EP 切分示意与各项通信量 |
| `<PagedKV :block-size="4" :num-blocks="24" />` | PagedAttention 的 block 分配、prefix 共享、swap/recompute 抢占 |
| `<ReleaseTimeline view="table" />` | vLLM 版本时间线：按类别筛选、点开看每版的 observation / 场景 / 做法；数据在 `src/data/vllm-releases.json` |

正文里可以用 `==重点==` 高亮，`==重点=={批注}` 加悬停批注（批注内支持行内 markdown）；`src/data/vllm-releases.json` 的文本字段也支持同样写法。读者在任意页面选中文字即可加自己的高亮 / 批注，右下角 ✎ 面板可导出导入。

计算逻辑在 `src/lib/`，组件只负责渲染。改公式请先改 `src/lib` 并补测试。

## 部署

推到 `main` 触发 `.github/workflows/deploy.yml`：跑测试 → 构建 → 部署到 Pages。
仓库设置里 Settings → Pages → Source 需选 **GitHub Actions**。

## 批注、闪卡在设备间同步

读者批注和闪卡的复习记录 / 批注 / 暂停存在 `src/data/` 下四个 json 里（清单见 `src/lib/syncDocs.ts`）。每次改动先存进浏览器，界面立刻更新、离线也不丢，再按下面的模式同步：

| 模式 | 什么时候 | 存到哪 |
|---|---|---|
| GitHub | 导航栏 ☁ 里连上了 GitHub token | 仓库的 `data` 分支，手机电脑在任何网络下共用 |
| 仓库文件 | 没连 token 的 `docs:dev` | 本机仓库里的 json，靠 git 同步 |
| 本机 | 没连 token 的线上站点 | 只在这台设备的浏览器里 |

- **连 GitHub**：点导航栏的 ☁，按提示建一个 fine-grained token，只授权这个仓库、Contents 选 Read and write。token 只存在那台设备上。
- **同步怎么做**：读远端 → 按条目合并（较新的胜出，删除留删除记录）→ 有变化就提交。停手 2 秒、切到后台、重新联网时同步一次，同一轮的几份数据合成一个 commit。提交时带上期望的分支 head，别的设备抢先提交了就重读重合并（`src/lib/sync.ts`、`src/lib/github.ts`）。
- **为什么是单独的 data 分支**：手机上每次同步都是一个 commit，不混进 main 的历史，不触发部署，电脑上 `git push` 也不会因此被拒。部署时 workflow 会把 data 分支上最新的四个文件拷进来，当站点的初始数据。连上 GitHub 之后，本机仓库里的这几个 json 不再更新，以 data 分支为准。
- **装到手机桌面**：站点是 PWA。iPhone 用 Safari 打开，分享 →「添加到主屏幕」；Android 用 Chrome 菜单里的「安装应用」。之后全屏打开，访问过的页面断网也能看（`docs/public/sw.js`）。

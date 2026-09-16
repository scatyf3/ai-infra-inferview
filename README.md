# AI Infra Inferview

北美 AI Infra 面试知识库。Markdown 文档 + 交互式可视化插件，部署在 GitHub Pages。

站点：<https://scatyf3.github.io/ai-infra-inferview/>

## 本地开发

```bash
npm install
npm run docs:dev      # http://localhost:5173/ai-infra-inferview/
npm run docs:build    # 生产构建（ignoreDeadLinks: false，坏链会失败）
npm run docs:preview  # 预览构建产物
npm test              # 计算逻辑的单元测试
```

## 结构

```
docs/                 VitePress 站点（srcDir）
  index.md            首页：交互式知识地图
  <domain>/*.md       7 个领域的知识文档
  .vitepress/
    domains.ts        领域元数据（唯一手工维护的结构信息）
    sidebar.ts        从 frontmatter 生成 sidebar
    data/topics.data.ts  构建时收集各页 status
    theme/components/    知识地图 + 四个可视化插件
src/lib/              纯 TS 计算逻辑（可单测，不依赖 Vue/DOM）
src/data/             GPU 与模型规格表
templates/topic.md    写作模板
```

## 写一篇新文档

复制 `templates/topic.md` 到对应领域目录，填 frontmatter：

```yaml
---
title: 中文标题（术语保留英文）
status: todo        # todo | draft | reviewed —— 驱动首页知识地图的节点颜色
tags: []
difficulty: 3       # 1–5，映射节点大小
order: 1            # 领域内排序
related: []         # 绝对路径，在地图上画虚线跨链
---
```

sidebar 和知识地图都从 frontmatter 自动生成，不需要改配置。`status` 写错值会让构建失败。

## 可视化插件

全局注册，直接在 Markdown 里顶格写标签：

| 组件 | 用途 |
|---|---|
| `<MemoryCalculator />` | 显存账 + roofline：输入模型/GPU/batch/context，输出权重 KV 激活拆解、需要几张卡、prefill 与 decode 的 AI 落点 |
| `<ShapeFlow variant="gqa" phase="decode" />` | 一层 Transformer 的 tensor shape 流转，可切 MHA/GQA/MQA/MLA 与 prefill/decode |
| `<ParallelismViz :tp="4" :pp="2" />` | TP/PP/DP/EP 切分示意与各项通信量 |
| `<PagedKV :block-size="4" :num-blocks="24" />` | PagedAttention 的 block 分配、prefix 共享、swap/recompute 抢占 |

计算逻辑在 `src/lib/`，组件只负责渲染。改公式请先改 `src/lib` 并补测试。

## 部署

推到 `main` 触发 `.github/workflows/deploy.yml`：跑测试 → 构建 → 部署到 Pages。
仓库设置里 Settings → Pages → Source 需选 **GitHub Actions**。

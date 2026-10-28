# handbook

手写推导的 LaTeX 版本。不在 `docs/` 下，VitePress 不会构建它。

## 结构

- `main.tex`：入口，按需 `\input` 各章节
- `chapters/`：每个推导一个 `.tex` 文件

## 编译

```bash
cd handbook
latexmk -xelatex main.tex
```

中文用 `ctexart`，需要 XeLaTeX。编译产物已在 `.gitignore` 里忽略。

// 原语闪卡：torch / triton 两组。直接改这个文件，页面自动更新。
//
// id 是复习记录的 key（src/data/flashcard-progress.json），改题面可以，改 id 会丢掉这张卡的记录。
// a 支持行内 `code`、**加粗**、$公式$ 和 \n 换行；qcode 是题面上的代码块（翻面前就显示），code 是答案下面的代码块；
// ref 是出处（站内链接，不含 base）。多行的代码用 lines`...` 写，可以跟着缩进。

import { lines, type Card } from '@lib/flashcards'
import { baguCards } from './flashcards-bagu'

const TP = '/handson/torch-primitives'
const TR = '/handson/triton_primitives'
const KM = '/handson/kernel-mindset'
const SM = '/handson/triton-softmax'

const primitiveCards: Card[] = [
  // ---------------- torch ----------------
  {
    id: 'torch-storage-index',
    deck: 'torch',
    topic: 'stride',
    q: '`x[i, j]` 在 storage 里是第几个元素？tensor 由哪几部分组成？',
    a: '`storage_offset + i*stride[0] + j*stride[1]`。tensor = 一维连续的 **storage** + 怎么看它的**元数据**（shape、stride、storage_offset，外加 dtype、device）。',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-contiguous-stride',
    deck: 'torch',
    topic: 'stride',
    q: 'shape `(a, b, c)` 的 tensor 连续（contiguous）时，stride 是多少？',
    a: '`(b*c, c, 1)`：每一维的 stride = 它右边所有维的 shape 之积，最后一维是 1。',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-transpose-meta',
    deck: 'torch',
    topic: 'stride',
    q: '`y = x.t()` 做了什么？`y` 和 `x` 是什么关系？',
    a: '只对调 shape 和 stride，**storage 不动**。`y` 和 `x` 共用同一块内存（`data_ptr` 相同），通过 `y` 写会改到 `x`；`y` 不再 contiguous。\n对调哪几维：`.t()` 只接受 ≤ 2 维的 tensor，2 维时对调的就是第 0、1 维，也就是仅有的两维。\n多维时：`transpose(d0, d1)` 指定对调哪两维；只换最后两维写 `transpose(-1, -2)`（或 `x.mT`）；`.T` 是把**全部维度倒过来**，多维用会有弃用警告。',
    code: 'x = torch.arange(6).view(2, 3)   # stride (3, 1)\ny = x.t()                        # shape (3, 2), stride (1, 3)\n\nz = torch.empty(4, 5, 6)\nz.transpose(-1, -2).shape        # (4, 6, 5)：只换最后两维\nz.T.shape                        # (6, 5, 4)：全部倒过来',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-view-vs-reshape',
    deck: 'torch',
    topic: 'view / reshape',
    q: '为什么 transpose 之后 `view` 会报错？`reshape` 和 `.contiguous()` 各做什么？',
    a: '`view` 的本质：新建一个 tensor 对象，指向**同一块 storage**，只换一套 shape 和 stride（必要时还有 offset），一个字节都不拷。所以它只能表达「按某组步长去读同一块内存」读得出来的形状，读不出来就报错。\n例子：attention 输出 `out` 是 `(h, N, dk)`、连续，stride `(N·dk, dk, 1)`。`transpose(0, 1)` 后 shape `(N, h, dk)`，stride `(dk, N·dk, 1)`。\n合成 `(N, h·dk)` 要把 h、dk 两维并成一维。相邻两维能合并的条件是 `stride[h] == shape[dk] × stride[dk]`，这里应该是 `dk`，实际是 `N·dk`：同一个 token 的各个 head 在内存里隔着 N·dk 个元素，不是连着的一段，一个 stride 表达不出来。\n`reshape`：能 view 就 view，不能就先拷一份连续的；`.contiguous()`：手动做这次拷贝。\n**默认用 `reshape`**；只有要通过结果写回原 tensor 时用 `view`（比如 `output.view(N, h, dk).copy_(y)`）：`view` 保证不拷，`reshape` 可能拷一份，写进副本也不报错。',
    code: 'out = torch.randn(h, N, dk)        # stride (N*dk, dk, 1)\nt = out.transpose(0, 1)            # (N, h, dk), stride (dk, N*dk, 1)\nt.view(N, h * dk)                  # RuntimeError: view size is not compatible ...\nt.reshape(N, h * dk)               # OK：内部先拷一份连续的\nt.contiguous().view(N, h * dk)     # 同上，手动拷',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-meta-vs-copy',
    deck: 'torch',
    topic: 'view / reshape',
    q: 'x 的 shape 是 `(1, 5)`。`x[:, 1:3]`、`x.t()`、`x.unsqueeze(0)`、`x.expand(4, -1)` 哪些和 x 共享内存？怎么验证？',
    a: '先看这四个操作各做什么（x 是 `(1, 5)`）：\n1. `x[:, 1:3]`：取第 1、2 列，shape `(1, 2)`。\n2. `x.t()`：转置，`(5, 1)`。\n3. `x.unsqueeze(0)`：在最前面插一个长度为 1 的维，`(1, 1, 5)`，数据不变。\n4. `x.expand(4, -1)`：把长度为 1 的第 0 维拉长到 4（`-1` 表示第 1 维不变），`(4, 5)`，4 行内容一样。\n**全部共享内存**：它们只改 shape、stride、storage_offset 这些元数据，不拷数据。\n验证：比较 `y.untyped_storage().data_ptr()` 和 x 的是否相同。不要比 `y.data_ptr()`：切片带 offset，首元素地址本来就不同。\n也可以改 y 看 x 变没变：切片、`.t()`、`unsqueeze` 的结果改了都会改到 x。`expand` 的结果多个位置指向同一块内存，不能原地写。',
    code: 'x = torch.arange(5.).view(1, 5)\nsame = lambda y: y.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()\nsame(x[:, 1:3]), same(x.t()), same(x.unsqueeze(0)), same(x.expand(4, -1))   # 全是 True\nx[:, 1:3].data_ptr() == x.data_ptr()                                        # False：带 offset',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-copy-when',
    deck: 'torch',
    topic: 'view / reshape',
    q: '`clone`、`contiguous()`、`reshape`、`repeat`、`cat` 哪些一定分配新内存，哪些看情况？',
    a: '**一定拷**：`clone`、`repeat`、`repeat_interleave`、`cat`，以及普通运算的结果（`x + 1`）。\n**看情况**：\n1. `contiguous()`：已经连续就直接返回自己，不连续才拷。\n2. `reshape`：能 view 就 view，不能才拷。\n所以 `reshape` 的结果可能和原 tensor 共享内存，也可能不共享，原地改它之前别假设。',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-expand-gqa',
    deck: 'torch',
    topic: 'view / reshape',
    q: 'GQA 里把 KV head 复制给多个 Q head，为什么 `expand` 比 `repeat_interleave` 省显存？',
    a: '`expand` 把被扩展维的 stride 设成 0，下标怎么变都读同一个元素，**不拷贝**；`repeat_interleave` 真的复制出一份新 tensor。',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-broadcast-rule',
    deck: 'torch',
    topic: 'broadcasting',
    q: '广播规则是什么？被广播的维度在内存里怎么实现？',
    a: '三步，核心是第 1 步：\n1. **维度数不够的，在左边补 1**：`(4,)` 和 `(3, 4)` 运算时，先把 `(4,)` 看成 `(1, 4)`。所以短的那个对上的永远是长的**最右边**几维。\n2. 从右往左逐维比较：要么相等，要么有一边是 1，否则报错。\n3. 结果每一维取较大的那个，长度为 1 的那边被「拉长」。\n实现：被拉长的维 stride 设成 0，和 `expand` 一样不拷贝。\n报错的例子：`(3, 4)` 加 `(3,)`。第 1 步补成 `(1, 3)`，第 2 步 4 对 3，报错。想按行加，要自己写成 `(3, 1)`。',
    code: 'a = torch.ones(3, 4)\na + torch.ones(4)       # (3, 4)：(4,) 补成 (1, 4)，每行加同一个向量\na + torch.ones(3)       # RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1\na + torch.ones(3, 1)    # (3, 4)：每行加一个数',
    ref: `${TP}#broadcasting`,
  },
  {
    id: 'torch-broadcast-trap',
    deck: 'torch',
    topic: 'broadcasting',
    q: '`(N,)` 加 `(N, 1)` 得到什么？怎么避免这类意外广播？',
    a: '得到 `(N, N)`，而且不报错。习惯：参与运算的 tensor 先用 `unsqueeze` / `x[:, None]` 把维度数补齐，让每一维的对应关系写在代码里，不靠右对齐规则去猜。',
    ref: `${TP}#容易踩的坑`,
  },
  {
    id: 'torch-keepdim',
    deck: 'torch',
    topic: 'broadcasting',
    q: '`logits (B, V) - logits.max(dim=-1).values` 有什么问题？',
    a: '想做的：每一行减去这一行的最大值。\n1. `logits.max(dim=-1).values` 的 shape 是 `(B,)`：被归约的那一维没了。\n2. 和 `(B, V)` 运算时，广播先在**左边**补 1，`(B,)` 变成 `(1, B)`，于是它对上的是 `(B, V)` 的**最后一维 V**，不是 B。\n3. 结果：B ≠ V 时报错；B = V 时不报错但算错。代码里第 0 行应该减 5、第 1 行减 3，实际变成第 0 列减 5、第 1 列减 3。\n修法：`keepdim=True` 让结果保留成 `(B, 1)`，补 1 补在右边那一维上，对上的才是行。规则：归约之后还要和原 tensor 运算的，一律加 `keepdim=True`。',
    code: lines`
      logits = torch.tensor([[1., 5.], [3., 2.]])         # B = V = 2
      m = logits.max(dim=-1).values                       # tensor([5., 3.])，shape (2,)
      logits - m                                          # [[-4., 2.], [-2., -1.]]  错：按列减了
      logits - logits.max(dim=-1, keepdim=True).values    # [[-4., 0.], [0., -1.]]   对：每行减自己的 max
    `,
    ref: `${TP}#容易踩的坑`,
  },
  {
    id: 'torch-matmul-batch',
    deck: 'torch',
    topic: 'operator',
    q: '`torch.matmul` 对哪些维度做矩阵乘、哪些维度广播？转置 K 为什么写 `transpose(-1, -2)` 而不是 `.T`？',
    a: '只对**最后两维**做矩阵乘，前面全当 batch 维并广播。`transpose(-1, -2)` 只换最后两维，前面有几个 batch 维都不用改；`.T` 会反转全部维度。',
    ref: `${TP}#矩阵乘`,
  },
  {
    id: 'torch-causal-triu',
    deck: 'torch',
    topic: 'mask',
    q: 'causal mask 的标准写法？为什么不用 `triu`？',
    a: '1. 本站模板：行下标 `i = arange(S)[:, None]`，列下标 `j = arange(S)[None, :]`，`allowed = j <= i`（True = 能看），`attn.masked_fill(~allowed, -inf)`。规则直接照公式写，不用推对角线编号。\n2. `triu` 版本只是备选：`full((S, S), -inf).triu(1)`。要记住 `k=1` 才不含主对角线；写成 `triu(0)` 会把对角线也盖掉，第一行全是 -inf，softmax 出 **NaN**。sliding window 这类规则要推两条对角线，最容易差一。\n3. 加法 mask 只能取交集；规则里有「或」（sink | window）只能用 bool `allowed`。',
    code: lines`
      i = torch.arange(S, device=x.device)[:, None]   # (S, 1) query
      j = torch.arange(S, device=x.device)[None, :]   # (1, S) key
      allowed = j <= i
      attn = attn.masked_fill(~allowed, float('-inf'))
    `,
    ref: '/leetgpu/mask',
  },
  {
    id: 'torch-masked-fill',
    deck: 'torch',
    topic: 'mask',
    q: '`masked_fill` 的语义？布尔版 causal mask 怎么写？',
    a: '1. `x.masked_fill(mask, value)` 的语义：mask 是 bool，**True 的位置换成 value**，False 的位置保留 x 原值；mask 先广播成 x 的 shape；返回新 tensor（原地版是 `masked_fill_`）。\n2. 本站约定先造 `allowed`（True = 能看）：行下标 `i = arange(S)[:, None]` 是 `(S, 1)`，列下标 `j = arange(S)[None, :]` 是 `(1, S)`，比较时广播成 `(S, S)`，`allowed[i, j]` 就是「query i 能不能看 key j」。causal 就是 `j <= i`。\n3. `~allowed` 是 bool 取反（逐元素 NOT）：True = **不能看**。所以 `masked_fill(~allowed, -inf)` 读作「不能看的位置填 -inf」。\n4. 放在 softmax **之前**：$e^{-\\infty} = 0$，这些位置的概率为 0。对角线 `j == i` 是 True，每行至少能看自己，不会整行 -inf 出 NaN。',
    fig: lines`
      S = 4, allowed = j <= i   (1 = True)
              j: 0 1 2 3
      i = 0      1 0 0 0
      i = 1      1 1 0 0
      i = 2      1 1 1 0
      i = 3      1 1 1 1
      ~allowed flips 1<->0 -> filled with -inf
    `,
    code: lines`
      i = torch.arange(S, device=x.device)[:, None]   # (S, 1) query 下标
      j = torch.arange(S, device=x.device)[None, :]   # (1, S) key 下标
      allowed = j <= i                                # (S, S)
      attn = attn.masked_fill(~allowed, float('-inf'))
      # 等价的旧写法：mask = ones(S, S, bool).triu(1) 直接表示「看不到」
    `,
    ref: '/leetgpu/mask',
  },
  {
    id: 'torch-full-device',
    deck: 'torch',
    topic: 'operator',
    q: '`torch.full` / `torch.ones` 构造 mask 时最常见的坑？`torch.empty` 呢？',
    a: '默认在 **CPU** 上，和 GPU tensor 运算报 device 不一致，一律传 `device=x.device`（或用 `x.new_zeros(shape)` 继承 dtype 和 device）。`torch.empty` 不初始化，读之前必须写满。',
    ref: `${TP}#构造-tensor`,
  },
  {
    id: 'torch-softmax-fn',
    deck: 'torch',
    topic: 'operator',
    q: '`torch.softmax(x, dim=-1)` 和 `nn.Softmax(x)` 有什么区别？要自己减 max 吗？',
    a: '`torch.softmax` 是函数；`nn.Softmax` 是 module 类，`nn.Softmax(x)` 是在构造对象。内部已经减过 max，数值稳定，不用自己减。',
    ref: `${TP}#归一化`,
  },
  {
    id: 'torch-unassigned-view',
    deck: 'torch',
    topic: 'modify',
    q: '单独一行 `Q.view(B, S, h, dk)` 有什么用？',
    a: '没用。`view` / `transpose` / `reshape` 都是 out-of-place，返回新 tensor，原来的 `Q` 不变；要写成 `Q = Q.view(...)`。',
    ref: `${TP}#容易踩的坑-1`,
  },
  {
    id: 'torch-output-copy',
    deck: 'torch',
    topic: 'modify',
    q: "LeetGPU 的 `solve(..., output)` 里写 `output = y` 为什么不对？",
    a: '只是让局部名字 `output` 指向别的 tensor，调用方那块显存没被写。要写进去：`output.copy_(y)`（src 会先广播成 dst 的 shape）。',
    ref: `${TP}#容易踩的坑-1`,
  },
  {
    id: 'torch-iadd',
    deck: 'torch',
    topic: 'modify',
    q: '`x += y` 和 `x = x + y` 一样吗？',
    a: '不一样。对 tensor，`x += y` 是**原地**写；`x` 是传进来的参数时会改到调用方。`x = x + y` 产生新 tensor，再让名字 `x` 指向它。',
    ref: `${TP}#容易踩的坑-1`,
  },
  {
    id: 'torch-inplace-training',
    deck: 'torch',
    topic: 'modify',
    q: '训练时为什么默认 out-of-place？哪些地方约定俗成用 in-place？',
    a: '手动 in-place 省不了多少显存（缓存分配器会复用、`torch.compile` 会融合），风险却是实的：改了反向要用的中间结果，会报 "modified by an inplace operation" 或梯度静默错。约定的 in-place：optimizer 更新参数、`zero_grad`、权重初始化（都在 `no_grad` 下），`ReLU(inplace=True)`；推理侧还有 KV cache、CUDA Graph 的预分配 buffer。',
    ref: `${TP}#modify-tensor`,
  },
  {
    id: 'torch-deterministic',
    deck: 'torch',
    topic: '可复现',
    q: '`torch.use_deterministic_algorithms(True)` 为什么会禁用一部分算子？',
    a: '这些算子内部用 `atomicAdd` 累加，先后顺序每次不同，而浮点加法不满足结合律，结果最后几位会变。要逐位可复现，就只能换成固定归约顺序的实现，或者直接报错。',
    ref: SM,
  },

  // ---------------- torch 原语：读代码（给代码和输入说输出）/ 写代码（给要做的事写代码） ----------------
  {
    id: 'torch-read-view',
    deck: 'torch',
    topic: 'view · 读代码',
    q: '读代码：每个表达式输出什么？（view）',
    qcode: lines`
      x = torch.arange(6)
      x.view(2, 3)
      x.view(3, -1)
    `,
    a: '`view(*shape)` 把同一块 storage 按行优先重新切成新 shape，元素总数要相等；`-1` 让 torch 自己算这一维（6 ÷ 3 = 2）。不拷贝，结果和 x 共享内存。',
    code: lines`
      >>> x.view(2, 3)
      tensor([[0, 1, 2],
              [3, 4, 5]])

      >>> x.view(3, -1)
      tensor([[0, 1],
              [2, 3],
              [4, 5]])
    `,
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-write-view',
    deck: 'torch',
    topic: 'view · 写代码',
    q: '写代码：`x` 是投影后的 Q，shape `(B, S, H*dk)`、连续（B batch，S 序列长度，H head 数，dk 每个 head 的维度）。想拆成每个 head 一份：`(B, S, H, dk)`，不拷贝。',
    a: '`x.view(B, S, H, dk)`。最后一维 H·dk 在内存里本来就连着，拆成 (H, dk) 只要换 shape 和 stride。',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-read-reshape',
    deck: 'torch',
    topic: 'reshape · 读代码',
    q: '读代码：每个表达式输出什么？（reshape）',
    qcode: lines`
      x = torch.arange(6).view(2, 3).t()
      x
      y = x.reshape(6)
      y
      y.data_ptr() == x.data_ptr()
    `,
    a: '`reshape` 按**逻辑顺序**（按当前 shape 行优先地读）排，不是按 storage 的顺序，所以是 0, 3, 1, 4, 2, 5。x 转置后不连续，展平没法只改 stride，reshape 就拷了一份：不共享内存。能 view 的时候它就等于 view。',
    code: lines`
      >>> x
      tensor([[0, 3],
              [1, 4],
              [2, 5]])

      >>> y
      tensor([0, 3, 1, 4, 2, 5])

      >>> y.data_ptr() == x.data_ptr()
      False
    `,
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-write-reshape',
    deck: 'torch',
    topic: 'reshape · 写代码',
    q: '写代码：attention 输出 `out` 的 shape `(N, H, dk)`，是从 `(H, N, dk)` transpose 来的、不连续（N token 数，H head 数，dk 每个 head 的维度）。想合并成 `(N, H*dk)`。',
    a: '`out.reshape(N, H * dk)`，或者 `out.contiguous().view(N, H * dk)`。直接 `view` 会报错：同一个 token 的各个 head 在内存里不相邻。',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-read-transpose',
    deck: 'torch',
    topic: 'transpose · 读代码',
    q: '读代码：每个表达式输出什么？（transpose）',
    qcode: lines`
      x = torch.arange(6).view(2, 3)
      y = x.transpose(0, 1)
      y
      y.stride()
    `,
    a: '`transpose(d0, d1)` 对调两个维度的 shape 和 stride，storage 不动：原来的 `x[i, j]` 就是 `y[j, i]`。x 的 stride 是 (3, 1)，对调成 (1, 3)。',
    code: lines`
      >>> y
      tensor([[0, 3],
              [1, 4],
              [2, 5]])

      >>> y.stride()
      (1, 3)
    `,
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-write-transpose',
    deck: 'torch',
    topic: 'transpose · 写代码',
    q: '写代码：q、k 的 shape 都是 `(B, H, S, dk)`（B batch，H head 数，S 序列长度，dk 每个 head 的维度）。想算 attention score `(B, H, S, S)`：第 i 行第 j 列是第 i 个 query 和第 j 个 key 的点积。',
    a: '`q @ k.transpose(-1, -2)`：k 换成 `(B, H, dk, S)`，matmul 对最后两维做 `(S, dk) @ (dk, S)`，B、H 当 batch。别用 `k.T`，它把四个维度全倒过来。',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-read-permute',
    deck: 'torch',
    topic: 'permute · 读代码',
    q: '读代码：每个表达式输出什么？（permute）',
    qcode: lines`
      x = torch.zeros(2, 3, 4)
      x.permute(2, 0, 1).shape
      x.permute(0, 2, 1).shape
      torch.arange(6).view(2, 3).permute(1, 0)
    `,
    a: '`permute(*dims)` 一次重排所有维度：结果的第 i 维是原来的第 `dims[i]` 维。所以 `permute(2, 0, 1)` 的 shape 是（原第 2 维, 原第 0 维, 原第 1 维）=（4, 2, 3）。只有两维时就是 transpose。',
    code: lines`
      >>> x.permute(2, 0, 1).shape
      torch.Size([4, 2, 3])

      >>> x.permute(0, 2, 1).shape
      torch.Size([2, 4, 3])

      >>> torch.arange(6).view(2, 3).permute(1, 0)
      tensor([[0, 3],
              [1, 4],
              [2, 5]])
    `,
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-write-permute',
    deck: 'torch',
    topic: 'permute · 写代码',
    q: '写代码：Q 拆完 head 后是 `(B, S, H, dk)`（B batch，S 序列长度，H head 数，dk 每个 head 的维度），attention 要的是 `(B, H, S, dk)`：每个 head 一个 `(S, dk)` 矩阵。',
    a: '`x.permute(0, 2, 1, 3)`，或者 `x.transpose(1, 2)`。都只改 stride，结果不连续；后面要 view 的话先 `contiguous()`。',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-read-unsqueeze',
    deck: 'torch',
    topic: 'unsqueeze · 读代码',
    q: '读代码：每个表达式输出什么？（unsqueeze）',
    qcode: lines`
      x = torch.tensor([1, 2, 3])
      x.unsqueeze(0)
      x.unsqueeze(1)
      x[:, None].shape
    `,
    a: '`unsqueeze(d)` 在第 d 个位置插一个长度为 1 的维，数据不动：(3,) 变成 (1, 3) 或 (3, 1)。`x[:, None]` 等于 `x.unsqueeze(1)`，`None` 写在哪就在哪插。',
    code: lines`
      >>> x.unsqueeze(0)
      tensor([[1, 2, 3]])

      >>> x.unsqueeze(1)
      tensor([[1],
              [2],
              [3]])

      >>> x[:, None].shape
      torch.Size([3, 1])
    `,
    ref: `${TP}#broadcasting`,
  },
  {
    id: 'torch-write-unsqueeze',
    deck: 'torch',
    topic: 'unsqueeze · 写代码',
    q: '写代码：`x` 的 shape `(B, D)`（B 行，每行 D 个数），每一行要乘自己的系数；系数 `s` 的 shape `(B,)`。',
    a: '`x * s.unsqueeze(1)`，或者 `x * s[:, None]`：s 变成 `(B, 1)`，广播时沿 D 拉长，第 b 行乘 s[b]。直接写 `x * s` 的话 s 对上的是 D 那一维：B ≠ D 报错，B = D 静默算错。',
    ref: `${TP}#broadcasting`,
  },
  {
    id: 'torch-read-squeeze',
    deck: 'torch',
    topic: 'squeeze · 读代码',
    q: '读代码：每个表达式输出什么？（squeeze）',
    qcode: lines`
      x = torch.zeros(1, 3, 1, 2)
      x.squeeze().shape
      x.squeeze(0).shape
      x.squeeze(1).shape
    `,
    a: '`squeeze()` 去掉**所有**长度为 1 的维。`squeeze(d)` 只去第 d 维，而且只在它长度为 1 时去，否则原样返回、不报错：第 1 维长度是 3，所以 shape 不变。',
    code: lines`
      >>> x.squeeze().shape
      torch.Size([3, 2])

      >>> x.squeeze(0).shape
      torch.Size([3, 1, 2])

      >>> x.squeeze(1).shape
      torch.Size([1, 3, 1, 2])
    `,
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-write-squeeze',
    deck: 'torch',
    topic: 'squeeze · 写代码',
    q: '写代码：模型最后一个位置的 logits 是 `(B, 1, V)`（B batch，V 词表大小），要变成 `(B, V)` 交给采样。B 可能等于 1。',
    a: '`logits.squeeze(1)`。别用 `squeeze()`：B = 1 时会把 batch 维也去掉，变成 `(V,)`。',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-read-expand',
    deck: 'torch',
    topic: 'expand · 读代码',
    q: '读代码：每个表达式输出什么？（expand）',
    qcode: lines`
      x = torch.tensor([[1], [2]])
      y = x.expand(2, 3)
      y
      y.stride()
    `,
    a: '`expand` 把长度为 1 的维「拉长」到给定长度，只能拉长长度为 1 的维。不拷贝：被拉长的那一维 stride 是 0，每一列读到的都是同一个元素。',
    code: lines`
      >>> y
      tensor([[1, 1, 1],
              [2, 2, 2]])

      >>> y.stride()
      (1, 0)
    `,
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-write-expand',
    deck: 'torch',
    topic: 'expand · 写代码',
    q: '写代码：布尔 causal mask `m` 的 shape `(S, S)`，某个 API 要求传 `(B, H, S, S)`（B batch，H head 数，S 序列长度），不想占额外显存。',
    a: '`m.expand(B, H, S, S)`：维度数不够时在左边补，新加的维 stride 为 0，不拷贝。别对结果原地写，好多位置指向同一块内存。',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-read-repeat',
    deck: 'torch',
    topic: 'repeat · 读代码',
    q: '读代码：每个表达式输出什么？（repeat）',
    qcode: lines`
      x = torch.tensor([1, 2])
      x.repeat(3)
      x.repeat(2, 2)
    `,
    a: '`repeat(*times)` 把**整个 tensor** 按每一维给的次数平铺：[1, 2] 重复 3 次是 [1, 2, 1, 2, 1, 2]。参数比维度多时，先在左边补长度 1 的维。真的拷贝。',
    code: lines`
      >>> x.repeat(3)
      tensor([1, 2, 1, 2, 1, 2])

      >>> x.repeat(2, 2)
      tensor([[1, 2, 1, 2],
              [1, 2, 1, 2]])
    `,
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-write-repeat',
    deck: 'torch',
    topic: 'repeat · 写代码',
    q: '写代码：`x = torch.tensor([[1, 2, 3], [4, 5, 6]])`，shape `(2, 3)`。想把它**整块**上下摞两份，得到 `(4, 3)`：前两行是 x，后两行还是 x。',
    a: '`x.repeat(2, 1)`。\n`repeat` 的参数是**每一维各铺几份**，有几维就写几个数，输出 shape = 输入 shape 逐维乘这些数：\n1. 第 0 维（行，往下数）铺 2 份：2 × 2 = 4 行。\n2. 第 1 维（列，往右数）铺 1 份，也就是不变：3 × 1 = 3 列。\n所以 `(2, 3)` → `(4, 3)`。「铺」是把**整块** x 当瓷砖往下贴，顺序是 x、x；想要每一行自己连着复制（行 0、行 0、行 1、行 1）用的是 `repeat_interleave`（见图）。',
    fig: lines`
      x rows: r0 = [1 2 3], r1 = [4 5 6]

      x.repeat(2, 1)                  r0 r1 r0 r1   whole block tiled
      x.repeat_interleave(2, dim=0)   r0 r0 r1 r1   each row copied in place
    `,
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-read-repeat-interleave',
    deck: 'torch',
    topic: 'repeat_interleave · 读代码',
    q: '读代码：每个表达式输出什么？（repeat_interleave）',
    qcode: lines`
      x = torch.tensor([[1, 2], [3, 4]])    # shape (2, 2)
      x.repeat_interleave(2, dim=0)
      x.repeat_interleave(2, dim=1)
    `,
    a: '先认维度：x 是 `(2, 2)`，第 0 维是**行**（往下数，r0 = [1, 2]、r1 = [3, 4]），第 1 维是**列**（往右数，c0 = [1, 3]、c1 = [2, 4]）。\n`repeat_interleave(n, dim)`：沿第 dim 维，把每一片**原地连着**复制 n 份。只有第 dim 维的长度乘 n，其他维不变。\n1. `dim=0`：切片是行，每行复制 2 份，顺序 r0 r0 r1 r1，`(2, 2)` → `(4, 2)`。\n2. `dim=1`：切片是列，每列复制 2 份，顺序 c0 c0 c1 c1，`(2, 2)` → `(2, 4)`。\n不写 dim 会先展平成一维再复制，得到 `[1, 1, 2, 2, 3, 3, 4, 4]`，所以一般都要写。',
    fig: lines`
      x (2, 2)     r0 = [1 2]
                   r1 = [3 4]

      dim=0 -> (4, 2)    dim=1 -> (2, 4)
        r0  [1 2]          c0 c0 c1 c1
        r0  [1 2]         [1  1  2  2]
        r1  [3 4]         [3  3  4  4]
        r1  [3 4]
    `,
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-write-repeat-interleave',
    deck: 'torch',
    topic: 'repeat_interleave · 写代码',
    q: '写代码：GQA 里 Q 有 4 个 head，K 只有 2 个 head：Q head 0、1 共用 K head 0，Q head 2、3 共用 K head 1。k 的 shape 是 `(2, S, dk)`，第 0 维是 head。想展开成 `(4, S, dk)`，让第 i 个位置正好是 Q head i 要用的 K。',
    a: '`k.repeat_interleave(2, dim=0)`。\n1. 复制几份：每个 K head 要给几个 Q head 用，叫**组大小** g = Q head 数 ÷ K head 数 = 4 ÷ 2 = 2。一般写法 `k.repeat_interleave(H // H_kv, dim=0)`。\n2. 沿哪一维：head 是第 0 维，所以 `dim=0`。\n3. 结果：每个 K head 原地连着复制 g 份，顺序 0, 0, 1, 1，第 i 个位置正好是 Q head i 要的（见图），shape `(2, S, dk)` → `(4, S, dk)`。\n易错：`k.repeat(2, 1, 1)` 是整块平铺，顺序 0, 1, 0, 1，Q head 1 拿到的是 K head 1，shape 对、不报错，但算错了。',
    fig: lines`
      Q head                      0  1  2  3
      repeat_interleave(2, dim=0) 0  0  1  1   right
      repeat(2, 1, 1)             0  1  0  1   wrong
    `,
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-read-cat',
    deck: 'torch',
    topic: 'cat · 读代码',
    q: '读代码：每个表达式输出什么？（cat）',
    qcode: lines`
      a = torch.tensor([[1, 2]])    # shape (1, 2)
      b = torch.tensor([[3, 4]])    # shape (1, 2)
      torch.cat([a, b], dim=0)
      torch.cat([a, b], dim=1)
    `,
    a: 'dim 是沿哪一维接，从左往右数、从 0 开始：(1, 2) 的第 0 维是行，第 1 维是列。\n1. `dim=0`：b 的行接在 a 的行下面，(1, 2) 和 (1, 2) 变成 (2, 2)。\n2. `dim=1`：b 的列接在 a 的列右边，变成 (1, 4)。\n只有被接的那一维长度相加，其他维必须一样长。',
    code: lines`
      >>> torch.cat([a, b], dim=0)
      tensor([[1, 2],
              [3, 4]])

      >>> torch.cat([a, b], dim=1)
      tensor([[1, 2, 3, 4]])
    `,
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-write-cat',
    deck: 'torch',
    topic: 'cat · 写代码',
    q: '写代码：没有预分配的 KV cache：`k_cache` 是 `(B, H, T, dk)`（T 已缓存的 token 数），这一步新算出的 `k_new` 是 `(B, H, 1, dk)`，接到最后。',
    a: '`k_cache = torch.cat([k_cache, k_new], dim=2)`。dim 从左往右数、从 0 开始：B 是 0，H 是 1，T 是 2，dk 是 3；要接在 token 那一维，所以是 2，结果 `(B, H, T + 1, dk)`。\n每步都要把整个 cache 拷一遍，所以推理框架用预分配 + 原地写，不用 cat。',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-read-stack',
    deck: 'torch',
    topic: 'stack · 读代码',
    q: '读代码：每个表达式输出什么？（stack）',
    qcode: lines`
      a = torch.tensor([1, 2, 3])    # shape (3,)
      b = torch.tensor([4, 5, 6])    # shape (3,)
      torch.stack([a, b], dim=0)
      torch.stack([a, b], dim=1)
    `,
    a: '`stack` 把 n 个同 shape 的 tensor 摞起来，**新插一维**，长度是 n（这里 n = 2）。dim 是新维插在第几个位置，不写默认 0。\n1. `dim=0`：新维插在最前，`(3,)` → `(2, 3)`。新维是行，第 i **行**是第 i 个 tensor。\n2. `dim=1`：新维插在最后，`(3,)` → `(3, 2)`。新维是列，第 i **列**是第 i 个 tensor。\n两个结果互为转置。对比 `cat`：cat 不插新维，只把已有的一维接长，`cat([a, b])` 是 `(6,)`。',
    code: lines`
      >>> torch.stack([a, b], dim=0)    # (2, 3)
      tensor([[1, 2, 3],
              [4, 5, 6]])

      >>> torch.stack([a, b], dim=1)    # (3, 2)
      tensor([[1, 4],
              [2, 5],
              [3, 6]])
    `,
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-write-stack',
    deck: 'torch',
    topic: 'stack · 写代码',
    q: '写代码：逐步 decode，每一步的输出 hidden 是 `(B, D)`，攒在列表 `outs` 里，一共 S 个。想得到 `(B, S, D)`。',
    a: '`torch.stack(outs, dim=1)`。用 `cat` 的话要先把每个变成 `(B, 1, D)`：`torch.cat([o.unsqueeze(1) for o in outs], dim=1)`。',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-read-matmul',
    deck: 'torch',
    topic: 'matmul / @ · 读代码',
    q: '读代码：每个表达式输出什么？（matmul / @）',
    qcode: lines`
      a = torch.tensor([[1, 2], [3, 4]])
      a @ torch.tensor([[1], [1]])
      (torch.zeros(2, 3, 4) @ torch.zeros(4, 5)).shape
      (torch.zeros(2, 1, 3, 4) @ torch.zeros(5, 4, 6)).shape
    `,
    a: '`@` 只对**最后两维**做矩阵乘 `(m, k) @ (k, n) → (m, n)`，前面的维都当 batch，按广播规则对齐：`(2, 1)` 和 `(5,)` 广播成 `(2, 5)`，所以最后是 `(2, 5, 3, 6)`。',
    code: lines`
      >>> a @ torch.tensor([[1], [1]])
      tensor([[3],
              [7]])

      >>> (torch.zeros(2, 3, 4) @ torch.zeros(4, 5)).shape
      torch.Size([2, 3, 5])

      >>> (torch.zeros(2, 1, 3, 4) @ torch.zeros(5, 4, 6)).shape
      torch.Size([2, 5, 3, 6])
    `,
    ref: `${TP}#矩阵乘`,
  },
  {
    id: 'torch-write-matmul',
    deck: 'torch',
    topic: 'matmul / @ · 写代码',
    q: '写代码：attention 权重 `p` 是 `(B, H, S, S)`（每行是一个 query 对所有 key 的概率），`v` 是 `(B, H, S, dk)`。求每个 query 对 v 的加权和。',
    a: '`p @ v`，shape `(B, H, S, dk)`：第 i 行 = Σ_j p[i, j] · v[j]。',
    ref: `${TP}#矩阵乘`,
  },
  {
    id: 'torch-read-softmax',
    deck: 'torch',
    topic: 'softmax · 读代码',
    q: '读代码：每个表达式输出什么？（softmax）',
    qcode: lines`
      x = torch.tensor([[0., 0.], [0., float('-inf')]])
      torch.softmax(x, dim=-1)
      torch.softmax(torch.full((2,), float('-inf')), dim=-1)
    `,
    a: '沿 dim 做 $\\text{softmax}(x)_i = e^{x_i} / \\sum_j e^{x_j}$，每一行（dim = -1）加起来是 1。两个相等的数各 0.5；$e^{-\\infty} = 0$，所以 -inf 的位置是 0。整行都是 -inf 时分母是 0，得到 NaN。',
    code: lines`
      >>> torch.softmax(x, dim=-1)
      tensor([[0.5000, 0.5000],
              [1.0000, 0.0000]])

      >>> torch.softmax(torch.full((2,), float('-inf')), dim=-1)
      tensor([nan, nan])
    `,
    ref: `${TP}#归一化`,
  },
  {
    id: 'torch-write-softmax',
    deck: 'torch',
    topic: 'softmax · 写代码',
    q: '写代码：attention score `s` 是 `(B, H, S, S)`，第 i 行是第 i 个 query 对所有 key 的分数。想把每个 query 的分数变成对 key 的概率分布。',
    a: '`torch.softmax(s, dim=-1)`：对最后一维（key）归一化，每行和为 1。写成 `dim=-2` 就成了对 query 归一化，错了也不报错。',
    ref: `${TP}#归一化`,
  },
  {
    id: 'torch-read-reduce',
    deck: 'torch',
    topic: 'max / sum（dim、keepdim） · 读代码',
    q: '读代码：每个表达式输出什么？（max / sum）',
    qcode: lines`
      x = torch.tensor([[1, 5], [3, 2]])
      x.max(dim=-1)
      x.sum(dim=0)
      x.sum(dim=-1, keepdim=True)
    `,
    a: '`dim` 是被**消掉**的那一维：`sum(dim=0)` 把各行加起来，每列一个结果（1 + 3，5 + 2）。`max(dim)` 同时返回最大值和它的下标。`keepdim=True` 把被消掉的维留成长度 1：(2, 2) 变成 (2, 1)，后面还要和 x 运算时才能按行对齐。',
    code: lines`
      >>> x.max(dim=-1)
      torch.return_types.max(
      values=tensor([5, 3]),
      indices=tensor([1, 0]))

      >>> x.sum(dim=0)
      tensor([4, 7])

      >>> x.sum(dim=-1, keepdim=True)
      tensor([[6],
              [5]])
    `,
    ref: `${TP}#容易踩的坑`,
  },
  {
    id: 'torch-write-reduce',
    deck: 'torch',
    topic: 'max / sum（dim、keepdim） · 写代码',
    q: '写代码：手写数值稳定的 softmax：`x` 是 `(B, V)`，每一行先减去这一行的最大值。',
    a: '`x - x.max(dim=-1, keepdim=True).values`\n1. `.values` 是什么：`x.max(dim=...)` 返回的不是一个 tensor，而是一对 `(values, indices)`：每行的最大值，和最大值在第几列（同 argmax）。`.values` 取前者。也可以用 `x.amax(dim=-1, keepdim=True)`，它只返回值。\n2. `dim=-1`：沿最后一维 V 求 max，每行一个数。\n3. `keepdim=True`：结果是 `(B, 1)` 而不是 `(B,)`，按行广播到 `(B, V)`。漏了 keepdim 就是 `(B,)`，广播时对上的是 V 那一维（见 keepdim 那张卡）。',
    code: lines`
      >>> x = torch.tensor([[1., 5.], [3., 2.]])
      >>> x.max(dim=-1)
      torch.return_types.max(
      values=tensor([5., 3.]),
      indices=tensor([1, 0]))
      >>> x - x.max(dim=-1, keepdim=True).values
      tensor([[-4.,  0.],
              [ 0., -1.]])
    `,
    ref: `${TP}#容易踩的坑`,
  },
  {
    id: 'torch-read-triu',
    deck: 'torch',
    topic: 'triu · 读代码',
    q: '读代码：每个表达式输出什么？（triu）',
    qcode: lines`
      torch.ones(3, 3).triu()
      torch.ones(3, 3).triu(1)
    `,
    a: '`triu(k)` 保留第 k 条对角线及它**右上方**的元素，其余置 0。k = 0 含主对角线；k = 1 从主对角线右边那条开始。',
    code: lines`
      >>> torch.ones(3, 3).triu()
      tensor([[1., 1., 1.],
              [0., 1., 1.],
              [0., 0., 1.]])

      >>> torch.ones(3, 3).triu(1)
      tensor([[0., 1., 1.],
              [0., 0., 1.],
              [0., 0., 0.]])
    `,
    ref: `${TP}#mask`,
  },
  {
    id: 'torch-write-triu',
    deck: 'torch',
    topic: 'mask · 写代码',
    q: '写代码：加法 causal mask：`(S, S)`，query i 只能看 key j ≤ i；看不到的位置是 -inf、能看到的是 0，加到 score 上。要和 score `x` 在同一个 device。',
    a: '分三步（代码见下）：\n1. 造 bool `allowed`：`i` 是 `(S, 1)` 的 query 下标，`j` 是 `(1, S)` 的 key 下标，`j <= i` 广播成 `(S, S)`，True = 能看。\n2. 转成加法 mask：全 0 的 `(S, S)` 上，把不能看（`~allowed`）的位置填 -inf。\n3. 加到 score 上：能看的位置 +0 不变，不能看的变成 -inf，softmax 后概率是 $e^{-\\infty} = 0$。\n为什么要加法 mask：它是一个普通的浮点 tensor，可以预先算好，直接作为 `attn_mask` 传给 `F.scaled_dot_product_attention` 这类接口。\n备选写法 `torch.full((S, S), float("-inf"), device=x.device).triu(1)` 结果一样；写成 `triu()`（k = 0）会把对角线也盖成 -inf，第 0 行全是 -inf，softmax 出 NaN。',
    code: lines`
      i = torch.arange(S, device=x.device)[:, None]
      j = torch.arange(S, device=x.device)[None, :]
      allowed = j <= i
      add_mask = torch.zeros(S, S, device=x.device).masked_fill(~allowed, float('-inf'))
      x = x + add_mask
    `,
    ref: '/leetgpu/mask',
  },
  {
    id: 'torch-read-masked-fill',
    deck: 'torch',
    topic: 'masked_fill · 读代码',
    q: '读代码：每个表达式输出什么？（masked_fill）',
    qcode: lines`
      x = torch.tensor([[1., 2.], [3., 4.]])
      m = torch.tensor([[False, True], [False, False]])
      x.masked_fill(m, 0.)
    `,
    a: '`x.masked_fill(mask, value)`：mask 为 True 的位置换成 value，其余保留 x 的值；mask 会广播成 x 的 shape；返回新 tensor。',
    code: lines`
      >>> x.masked_fill(m, 0.)
      tensor([[1., 0.],
              [3., 4.]])
    `,
    ref: `${TP}#mask`,
  },
  {
    id: 'torch-write-masked-fill',
    deck: 'torch',
    topic: 'masked_fill · 写代码',
    q: '写代码：padding mask：`pad` 是 `(B, S)`，`True` 表示这个位置是补出来的 padding。score `s` 是 `(B, H, S, S)`（最后一维是 key），任何 query 都不能看 padding 的 key。再叠上 causal 呢？',
    a: '1. 为什么只挡 key：s 的第 i 行、第 j 列是「query i 看 key j」的分数。padding 的 key 是假 token，任何 query 都不该看它，所以挡的是**列**，也就是最后一维。padding 的 query 不用挡：它这一行的输出最后会被丢掉（不算 loss、不返回）。要是连它的行也全挡成 -inf，这一行 softmax 是 0 / 0，出 NaN，反而会污染后面的计算。\n2. 怎么对齐维度：s 是 `(B, H, S_q, S_k)`，pad 是 `(B, S)`，要让 pad 的 S 落在最后一维 `S_k` 上。`pad[:, None, None, :]` 在中间插两个长度 1 的维，变成 `(B, 1, 1, S)`，广播到每个 head、每个 query。`allowed = ~pad[...]`：不是 padding 才能看。\n3. 叠 causal：再 `&` 一条规则 `j <= i`，`(S, S)` 和 `(B, 1, 1, S)` 广播成 `(B, 1, S, S)`。\n4. 最后统一 `masked_fill(~allowed, -inf)` 一次。',
    code: lines`
      i = torch.arange(S, device=s.device)[:, None]   # (S, 1) query
      j = torch.arange(S, device=s.device)[None, :]   # (1, S) key
      allowed = ~pad[:, None, None, :]                # (B, 1, 1, S)
      allowed = allowed & (j <= i)                    # (B, 1, S, S)
      s = s.masked_fill(~allowed, float('-inf'))
    `,
    ref: '/leetgpu/mask',
  },
  {
    id: 'torch-read-full',
    deck: 'torch',
    topic: 'full · 读代码',
    q: '读代码：每个表达式输出什么？（full）',
    qcode: lines`
      torch.full((2, 3), 7)
      torch.full((2,), 7).dtype
      torch.full((2,), 0.5).dtype
    `,
    a: '`torch.full(shape, value)` 造一个全是 value 的 tensor。dtype 从 value 推：整数是 int64，浮点是 float32。不传 device 就在 CPU 上。',
    code: lines`
      >>> torch.full((2, 3), 7)
      tensor([[7, 7, 7],
              [7, 7, 7]])

      >>> torch.full((2,), 7).dtype
      torch.int64

      >>> torch.full((2,), 0.5).dtype
      torch.float32
    `,
    ref: `${TP}#构造-tensor`,
  },
  {
    id: 'torch-write-full',
    deck: 'torch',
    topic: 'full · 写代码',
    q: '写代码：decode 一步的 attention 要分块遍历 KV cache，用 online softmax 维护每个 query 的运行最大值 `m`。`q` 是 `(B, H, dk)`（每个序列、每个 head 只有 1 个新 token）。`m` 是什么 shape？初值全是 -inf，和 `q` 同 dtype、同 device。',
    a: '`m = torch.full((B, H), float(\'-inf\'), dtype=q.dtype, device=q.device)`\n1. 为什么是 `(B, H)`：online softmax 给**每一行 score** 维护一个运行最大值，一行 score 对应一个 query。decode 时每个序列、每个 head 只有 1 个 query，一共 B × H 行，所以是 `(B, H)`。prefill 时每个 head 有 S 个 query，就是 `(B, H, S)`。\n2. 为什么初值是 -inf：它是 max 的单位元，`max(-inf, x) = x`，第一块进来就会被换掉。\n3. 为什么要传 device：不传就建在 CPU 上，和 GPU 上的 q 一运算就报 device 不一致。dtype 同理。',
    ref: `${TP}#构造-tensor`,
  },
  {
    id: 'torch-read-copy',
    deck: 'torch',
    topic: 'copy_ · 读代码',
    q: '读代码：每个表达式输出什么？（copy_）',
    qcode: lines`
      out = torch.zeros(2, 3)
      out.copy_(torch.tensor([1., 2., 3.]))
      out[0].copy_(torch.tensor([7., 8., 9.]))
      out
    `,
    a: '`dst.copy_(src)` 把 src 的值原地写进 dst：src 先广播成 dst 的 shape，dst 的 shape 不变，返回 dst。往 view（`out[0]`）里写也会写到 out 上。',
    code: lines`
      >>> out.copy_(torch.tensor([1., 2., 3.]))
      tensor([[1., 2., 3.],
              [1., 2., 3.]])

      >>> out[0].copy_(torch.tensor([7., 8., 9.]))
      tensor([7., 8., 9.])

      >>> out
      tensor([[7., 8., 9.],
              [1., 2., 3.]])
    `,
    ref: `${TP}#modify-tensor`,
  },
  {
    id: 'torch-write-copy',
    deck: 'torch',
    topic: 'copy_ · 写代码',
    q: '写代码：LeetGPU 风格：`def solve(x, output)`，`output` 是调用方预先分好的 tensor，shape 和结果一样。算出结果 `y` 后要交给调用方。',
    a: '`output.copy_(y)`。写成 `output = y` 只是让函数里的名字 output 指向 y，调用方那块显存没变。',
    ref: `${TP}#modify-tensor`,
  },
  {
    id: 'torch-read-where',
    deck: 'torch',
    topic: 'where · 读代码',
    q: '读代码：每个表达式输出什么？（where）',
    qcode: lines`
      x = torch.tensor([-1., 2., -3.])
      torch.where(x > 0, x, 0.)
      torch.where(x > 0)
    `,
    a: '`torch.where(cond, a, b)`：cond 为 True 取 a，否则取 b，三者按广播对齐，b 可以是标量。只传 cond 时返回 True 的位置的下标（tuple，每一维一个 tensor）。',
    code: lines`
      >>> torch.where(x > 0, x, 0.)
      tensor([0., 2., 0.])

      >>> torch.where(x > 0)
      (tensor([1]),)
    `,
    ref: `${TP}#broadcasting`,
  },
  {
    id: 'torch-write-where',
    deck: 'torch',
    topic: 'where · 写代码',
    q: '写代码：不用 `relu`，把 `x` 里的负数都换成 0，正数保留。',
    a: '`torch.where(x > 0, x, 0.)`，或者 `x.clamp(min=0)`。',
    ref: `${TP}#broadcasting`,
  },
  {
    id: 'torch-read-contiguous',
    deck: 'torch',
    topic: 'contiguous · 读代码',
    q: '读代码：每个表达式输出什么？（contiguous）',
    qcode: lines`
      x = torch.arange(6).view(2, 3).t()
      x.is_contiguous()
      y = x.contiguous()
      y.stride()
    `,
    a: '`contiguous()` 返回按当前 shape 行优先连续存放的 tensor。x 转置后不连续，所以拷了一份，stride 变成 (2, 1)。已经连续的 tensor 调它不拷，直接返回自己。',
    code: lines`
      >>> x.is_contiguous()
      False

      >>> y.stride()
      (2, 1)
    `,
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-write-contiguous',
    deck: 'torch',
    topic: 'contiguous · 写代码',
    q: '写代码：`x` 是 transpose 出来的、不连续。要把它传给一个按 `ptr + i` 线性读内存的自定义 CUDA kernel。',
    a: '先 `x = x.contiguous()` 再传指针。不连续时按 `ptr + i` 读到的是 storage 的顺序，不是 x 逻辑上的顺序，结果静默错。',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-read-clone',
    deck: 'torch',
    topic: 'clone · 读代码',
    q: '读代码：每个表达式输出什么？（clone）',
    qcode: lines`
      x = torch.tensor([1, 2, 3])
      y = x.clone()
      z = x[:]
      y[0] = 100
      z[1] = 200
      x
    `,
    a: '`clone()` 拷一份新内存，改 y 不影响 x；切片 `x[:]` 是 view，和 x 共用内存，改 z 就改了 x。',
    code: lines`
      >>> x
      tensor([  1, 200,   3])
    `,
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-write-clone',
    deck: 'torch',
    topic: 'clone · 写代码',
    q: '写代码：beam search 里要给当前的 KV 张量 `kv` 存一份快照；之后 `kv` 会被原地改，快照不能跟着变。',
    a: '`snapshot = kv.clone()`。`kv[:]`、`kv.view(...)` 都和 kv 共用内存，会跟着变。还要断开梯度的话加 `.detach()`。',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },

  // ---------------- triton ----------------
  {
    id: 'triton-program-vs-thread',
    deck: 'triton',
    topic: '编程模型',
    q: 'Triton 和 CUDA 的编程单位有什么不同？`threadIdx` 对应什么？',
    a: 'Triton 只写到 **program**（≈ CUDA 的 block）这一级，一个 program 一次处理一整块数据，写起来像操作一个小 tensor。**没有 threadIdx**：program 用 `num_warps × 32` 个线程执行，每个线程拿哪几个元素由编译器决定。',
    ref: `${TR}#映射-program-而不是-thread`,
  },
  {
    id: 'triton-offs-mask',
    deck: 'triton',
    topic: 'offs / mask',
    q: '1D kernel 开头那三行怎么写？分别是什么？',
    a: '`pid` 是我是第几个 program（≈ `blockIdx.x`）；`offs` 是这个 program 负责的**下标向量**；`mask` 挡掉最后一块越界的部分，传给 load / store。',
    code: 'pid = tl.program_id(0)\noffs = pid * BLOCK + tl.arange(0, BLOCK)\nmask = offs < n',
    ref: TR,
  },
  {
    id: 'triton-grid-cdiv',
    deck: 'triton',
    topic: 'offs / mask',
    q: 'n = 1000 个元素、BLOCK = 256，grid 怎么写？要开几个 program？为什么向上取整？',
    a: '`grid = (triton.cdiv(n, BLOCK),)`。`cdiv(n, BLOCK)` 就是**块数**，也就是 program 个数，相当于 CUDA 的 `gridDim.x`；grid 是 tuple，每个元素是那一维的块数。\n1. 向上取整：`cdiv(a, b) = (a + b - 1) // b`，`cdiv(1000, 256) = 4`。\n2. 前 3 块是满的，共 768 个元素；第 4 块的 `offs` 是 768..1023，只有 768..999 有效，剩下 24 个用 `mask = offs < n` 挡掉。\n3. 向下取整只有 3 块，最后 232 个元素没有 program 负责，输出是垃圾值。',
    ref: TR,
  },
  {
    id: 'triton-constexpr',
    deck: 'triton',
    topic: 'constexpr',
    q: '`BLOCK` 为什么要标 `tl.constexpr`？kernel 里直接用外面的 Python 变量 `BLOCK_SIZE` 会怎样？',
    a: '`BLOCK` 决定 `tl.arange` 的长度和寄存器分配，必须编译期确定；换一个值就重新编译一份 kernel。直接引用外面的普通 Python 变量会报错，要放进参数并标 `BLOCK_SIZE: tl.constexpr`。',
    ref: TR,
  },
  {
    id: 'triton-arange-pow2',
    deck: 'triton',
    topic: 'constexpr',
    q: '`NB_BLOCK = max(16, triton.next_power_of_2(NB))` 是在干嘛？',
    a: '`tl.arange` 的长度**必须是 2 的幂**。要一次读进 NB = 123 个局部结果，就补到 128，多出来的用 `i < NB` 挡掉。`max(16, ...)` 是给 N 很小、NB = 1 时兜底。',
    ref: SM,
  },
  {
    id: 'triton-load-other',
    deck: 'triton',
    topic: 'offs / mask',
    q: '`tl.load` 带 mask 时 `other` 不写会怎样？求 max 和求 sum 时各填什么？',
    a: '被 mask 的位置值**未定义**（常常是 0），全是负数时 max 会被拉成 0。填对运算没有影响的值：求 max 填 `-inf`，求 sum 填 `0`。softmax 读 x 填 `-inf` 两边都对：`exp(-inf) = 0`。',
    code: "x = tl.load(x_ptr + offs, mask=offs < N, other=-float('inf'))",
    ref: SM,
  },
  {
    id: 'triton-load-scalar',
    deck: 'triton',
    topic: 'load / store',
    q: '`tl.load(m_ptr)` 读到的是什么？要读 `m` 的前 NB 个元素怎么写？',
    a: '只读指针指向的**一个**数（`m[0]`）。读一段要给指针向量：',
    code: "i = tl.arange(0, NB_BLOCK)\nms = tl.load(m_ptr + i, mask=i < NB, other=-float('inf'))",
    ref: SM,
  },
  {
    id: 'triton-store-shape',
    deck: 'triton',
    topic: 'load / store',
    q: '`tl.store` 有 axis 参数吗？指针和值的形状要满足什么？',
    a: '没有 axis，它只管写到哪、写什么。指针和值形状对应：一个地址配一个标量（`tl.store(d + pid, s)`），指针向量配同长度的值向量，还要传 `mask`。',
    ref: SM,
  },
  {
    id: 'triton-reduce-axis',
    deck: 'triton',
    topic: '归约',
    q: '`tl.sum` / `tl.max` 一定要写 axis 吗？',
    a: 'Triton 3.x 默认 `axis=None`，全部维度归约成一个标量，1D 时和 `axis=0` 一样。2D 时必须想清楚沿哪一维；旧版本要求必须写。习惯写上 `axis=0`。',
    ref: SM,
  },
  {
    id: 'triton-exp-xor',
    deck: 'triton',
    topic: '归约',
    q: '在 kernel 里写 `e^x` 和 `tl.reduce(...)` 求和有什么问题？',
    a: 'Python 的 `^` 是按位**异或**，求 e 的幂用 `tl.exp(x)`。`tl.reduce` 要自己传合并函数，求和直接用 `tl.sum(x, axis=0)`。',
    ref: SM,
  },
  {
    id: 'triton-2d-ptrs',
    deck: 'triton',
    topic: '2D 指针',
    q: 'A 是行主序的 `(M, K) = (4, 8)` 矩阵，stride 是 `(8, 1)`。一个 program 要读左上角 2 × 4 的 tile，指针矩阵怎么拼？元素偏移各是多少？',
    a: 'Triton 不认 shape，只认指针：元素 `A[r][c]` 的地址是 `base + r * stride_m + c * stride_k`，行主序下 `stride_m = K = 8`，`stride_k = 1`。\n1. 行下标 `offs_m = [0, 1]`，列下标 `offs_k = [0, 1, 2, 3]`。\n2. `offs_m[:, None]` 是 `(2, 1)`，`offs_k[None, :]` 是 `(1, 4)`，相加时广播成 `(2, 4)`，规则和 torch 一样。\n3. 偏移矩阵是 `[[0, 1, 2, 3], [8, 9, 10, 11]]`：每行内部连续，换一行跳 8 个元素。\n边界：tile 可能伸出矩阵，mask 也用同样的广播拼：`(offs_m[:, None] < M) & (offs_k[None, :] < K)`。',
    code: lines`
      offs_m = pid_m * BM + tl.arange(0, BM)        # (BM,)
      offs_k = tl.arange(0, BK)                     # (BK,)
      ptrs = a_ptr + offs_m[:, None] * stride_m + offs_k[None, :] * stride_k   # (BM, BK)
      mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
      a = tl.load(ptrs, mask=mask, other=0.0)
    `,
    ref: TR,
  },
  {
    id: 'triton-you-vs-compiler',
    deck: 'triton',
    topic: '编程模型',
    q: '写 Triton kernel 时，哪些由你决定、哪些由编译器决定？',
    a: '**你**：一个 program 负责哪块数据（grid、`BLOCK`）、`num_warps`、`num_stages`。**编译器**：块内每个线程拿哪几个元素（layout）、合并访存、何时放进 shared memory、何时同步。',
    ref: `${TR}#你决定-vs-编译器决定`,
  },
  {
    id: 'triton-num-warps',
    deck: 'triton',
    topic: '编程模型',
    q: '一个按行做 softmax 的 kernel，一个 program 处理一行，BLOCK = 1024，`num_warps = 4` 时每个线程负责几个元素？BLOCK = 16384 呢？`num_warps` 该怎么调？',
    a: '`num_warps` 是一个 program 用几个 warp，一个 warp 32 个线程。编译器把 BLOCK 个元素平均分给这些线程：每线程 = BLOCK / (32 × num_warps)。\n1. BLOCK = 1024、4 个 warp（128 个线程）：每线程 **8** 个。\n2. BLOCK = 16384、4 个 warp：每线程 **128** 个 fp32，一个数组就占 128 个寄存器，softmax 还要存 x、exp 等中间值，会超过每线程 255 个寄存器的上限，溢出到 local memory，变慢。改成 16 个 warp（512 个线程）就是每线程 32 个。\n3. 反过来，块很小时 warp 开多了，每个线程只分到一两个元素，大量线程闲着，还多了 warp 之间的同步。\n经验：每线程几个到几十个元素比较合适；块越大，warp 越多。最终还是用 `triton.autotune` 实测选。',
    ref: SM,
  },
  {
    id: 'triton-no-grid-sync',
    deck: 'triton',
    topic: '跨 program',
    q: 'N = 500k 的 softmax 分成 123 块，每块先算出自己的 max，第 2 步要用全局 max M。为什么不能在同一个 kernel 里让每个 program「等所有块都算完」再继续，而要切一个新 kernel？CUDA 里有办法吗？',
    a: '1. Triton 里不同 program 之间**没有全局同步**原语。能保证「上一步所有 program 都写完了」的只有 kernel 边界：同一个 stream 里，后一个 kernel 一定在前一个全部结束后才开始。\n2. 自己写忙等（比如用 atomic 计数器数到 123 再继续）会**死锁**：GPU 不保证 123 个 program 同时在跑。如果先上去的 program 占满了 SM、原地等，还没上去的 program 永远排不上，计数永远到不了 123。\n3. CUDA 一样：block 内有 `__syncthreads()`；跨 block 的普通 launch 也没有。cooperative groups 有 `cg::this_grid().sync()`，但要用 `cudaLaunchCooperativeKernel` 启动，而且 grid 不能超过「SM 数 × 每个 SM 能同时放的 block 数」，保证所有 block 同时驻留，这正是第 2 点死锁的解法。限制多，所以实际也常拆 kernel 或用 atomic 合并。\n4. program 内部的同步由 Triton 编译器自动插，不用自己写（`tl.debug_barrier()` 只用于调试）。',
    ref: SM,
  },
  {
    id: 'triton-stream-order',
    deck: 'triton',
    topic: '跨 program',
    q: '连续 launch 三个有依赖的 kernel，要手动同步吗？Python 里 launch 那一行会等 GPU 算完吗？',
    a: '不用。同一个 CUDA stream 里的 kernel **按提交顺序**执行，前一个跑完后一个才开始。launch 是**异步**的，只是提交到 stream 就返回。\n1. 这不是「自动分析依赖」：GPU 不看谁读写了什么，只是同一个 stream 串行执行，所以有依赖的 kernel 放在同一个 stream 里就安全。放在**不同 stream** 就没有先后保证，要用 event 显式等（`cudaStreamWaitEvent` / `stream.wait_event`）。\n2. CUDA 也一样：`kernel<<<...>>>` 是异步提交，同一个 stream 自动排队，两个 kernel 之间**不用**手动 sync。只有 CPU 要读结果时才同步：`cudaDeviceSynchronize()` / `cudaStreamSynchronize()`；同步版的 `cudaMemcpy` 拷回 host 时也会隐式等。\n3. PyTorch 里 `.item()`、`.cpu()`、`print(tensor)` 会隐式同步；计时前后要 `torch.cuda.synchronize()`，否则量到的只是提交的时间。',
    ref: SM,
  },
  {
    id: 'triton-softmax-largeN',
    deck: 'triton',
    topic: 'softmax',
    q: 'N = 500k 的 softmax，「先 max 再 sum」和 online 合并各要几个 kernel、读几遍 x？',
    a: '先 max 再 sum：**3** 个 kernel（局部 max → 用全局 M 求局部 sum → 归一化），读 x **3** 遍。online：**2** 个 kernel（局部 `(m_b, d_b)` → 合并后归一化），读 **2** 遍。\n选哪个：一般选 **online**，少读一遍、少一次 launch。数据超出 L2 时，多读的那一遍是实打实的 HBM 流量，差距更明显；FlashAttention 里 score 根本不写回 HBM，只能用 online。\n这道题 x 只有 2MB，放得进 L2，两种的实测差距主要就是一次 launch（几 µs），都能过。面试手写时可以先写三趟（好讲、不容易错），再说 online 怎么省掉一趟。',
    ref: SM,
  },
  {
    id: 'triton-online-merge',
    deck: 'triton',
    topic: 'softmax',
    q: '已知每块的 $(m_b, d_b)$，全局的 M、D 怎么得到？代码里 M、D 放在哪？',
    a: '每块先只看自己：\n$$m_b = \\max_{i \\in b} x_i, \\qquad d_b = \\sum_{i \\in b} e^{x_i - m_b}$$\n合并：\n$$M = \\max_b m_b, \\qquad D = \\sum_b d_b \\, e^{m_b - M}$$\n每块的和是相对 $m_b$ 算的，乘 $e^{m_b - M}$ 换算成相对 M；因为 $m_b \\le M$，修正因子 ≤ 1，不会溢出。\nM、D 怎么分配：\n1. 显存里只分配每块一格的 `m`、`d`：`torch.empty(NB)`，NB 是块数。不用初始化，第 1 趟每格都会写。\n2. M、D **不占显存**：第 2 趟每个 program 都把 NB 个 `m`、`d` 读回来，在寄存器里自己算一遍 M、D，再归一化自己那块。',
    code: lines`
      def solve(x, out, N):
          BLOCK = 4096
          NB = triton.cdiv(N, BLOCK)                          # 块数
          NB_BLOCK = max(16, triton.next_power_of_2(NB))      # tl.arange 的长度要是 2 的幂
          m = torch.empty(NB, device=x.device, dtype=torch.float32)   # 每块一格
          d = torch.empty_like(m)
          partial_kernel[(NB,)](x, m, d, N, BLOCK=BLOCK)
          norm_kernel[(NB,)](x, out, m, d, N, NB, BLOCK=BLOCK, NB_BLOCK=NB_BLOCK)

      # 第 1 趟 partial_kernel：每块算 (m_b, d_b)，写进自己那一格
      m_b = tl.max(x, axis=0)
      tl.store(m_ptr + pid, m_b)
      tl.store(d_ptr + pid, tl.sum(tl.exp(x - m_b), axis=0))

      # 第 2 趟 norm_kernel：读回所有块，在寄存器里合并出 M、D
      i = tl.arange(0, NB_BLOCK)
      ms = tl.load(m_ptr + i, mask=i < NB, other=-float('inf'))
      ds = tl.load(d_ptr + i, mask=i < NB, other=0.0)        # 补齐的位置：0 * exp(-inf) = 0
      M = tl.max(ms, axis=0)
      D = tl.sum(ds * tl.exp(ms - M), axis=0)
    `,
    ref: `${SM}#大-n-先-max-再-sum-vs-online-合并`,
  },
  {
    id: 'triton-why-one-program-no',
    deck: 'triton',
    topic: 'softmax',
    q: 'N = 500k，为什么不让一个 program 包下整个向量？',
    a: '单个 block 上限 2^20 个元素，从限制上放得下，但只有 **1 个 SM** 在干活，其余一百多个闲着；这么多数据也塞不进寄存器，会溢出到 local memory。',
    ref: SM,
  },
  {
    id: 'triton-atomic-init',
    deck: 'triton',
    topic: 'atomic',
    q: '用 `tl.atomic_max` / `tl.atomic_add` 合并全局 M、D 时，初值怎么给？结果可复现吗？',
    a: '初值要取**单位元**：和任何数合并都不改变那个数。\n1. max 的单位元是 −inf：`max(−inf, x) = x`。所以 M 用 `torch.full((1,), float("-inf"))`。\n2. 加法的单位元是 0：`0 + x = x`。所以 D 用 `torch.zeros(1)`。注意这里的 `(1)` 是 **shape**，意思是只有一个元素，值是 **0** 不是 1；写成 `torch.full((1,), 0.0)` 更不容易看错。\n3. **不能用 `torch.empty`**：它的内容是显存里残留的任意值，合并进去结果就错了。\n可复现吗：max 和合并顺序无关，每次一样；`atomic_add` 的先后顺序每次不同，浮点加法不满足结合律，最后几位会变。',
    code: 'M = torch.full((1,), float("-inf"), device="cuda")   # max 的单位元\nD = torch.zeros(1, device="cuda")                     # shape (1,)，值是 0\nD = torch.full((1,), 0.0, device="cuda")              # 同上，写法更直观',
    ref: SM,
  },
  {
    id: 'triton-atomic-cost',
    deck: 'triton',
    topic: 'atomic',
    q: 'atomic 什么时候慢？softmax 里每块做一次 atomic 有问题吗？',
    a: '慢在**很多线程对同一个地址做 atomic**：硬件只能一个接一个地处理，没法并行。\n1. 为什么串行：global atomic 不在 SM 里算，而是发到 L2 cache，由那里的原子单元做「读 → 改 → 写」。同一个地址的第 2 次加法必须看到第 1 次的结果，只能排队；不同地址落在不同的 L2 分片上，可以同时处理。\n2. 逐元素 atomic：N = 50 万个元素都加到同一个 D 上，就是 50 万次排队，很慢。\n3. 每块一次：先在块内 `tl.sum` 归约，每块只做 1 次 atomic，NB = 123 块就只有 123 次，可以忽略。\n这时真正多出来的开销是：D 要先初始化成 0，多一次 kernel launch。',
    ref: SM,
  },
  {
    id: 'triton-partial-deterministic',
    deck: 'triton',
    topic: 'atomic',
    q: '局部结果数组（每块写一格、再合并）为什么能逐位复现？',
    a: '加法顺序固定：块内 `tl.sum` 的归约树是编译期定的，第 b 块固定写 `d[b]`，合并也是固定顺序。同 GPU、同 Triton 版本、同 BLOCK / num_warps 下每次一样。',
    ref: SM,
  },
  {
    id: 'triton-matmul-program',
    deck: 'triton',
    topic: 'matmul',
    q: 'Triton matmul `C (M×N) = A (M×K) · B (K×N)`，M = N = K = 4096，BM = BN = 128，BK = 32，bf16。一个 program 负责什么？grid 多大？K 维循环几次？每个 tile 的算术强度是多少？',
    a: '一个 program 负责 C 的一个 `(BM, BN)` tile：M、N 是并行维，进 grid；K 是归约维，在 program 里循环（代码见下）。\n1. grid：`(4096 / 128) × (4096 / 128) = 32 × 32 = 1024` 个 program。\n2. K 循环：4096 / 32 = **128** 次。每次读 A 的 `(128, 32)` 和 B 的 `(32, 128)`，各 128 × 32 × 2 B = 8 KiB，做 `acc += tl.dot(a, b)`。\n3. `acc` 是 `(128, 128)` 的 fp32 累加器，留在寄存器里，循环结束才写回一次。\n4. 算术强度：每次循环做 BM·BN·BK 次乘加 = 2·BM·BN·BK FLOP，读 (BM + BN)·BK·2 字节，相除得 $\\frac{2 \\cdot BM \\cdot BN}{(BM + BN) \\cdot 2} = \\frac{BM \\cdot BN}{BM + BN} = 64$ FLOP/B。tile 越大越高，这就是 GEMM 能 compute-bound、GEMV（BM = batch 很小）不能的原因。',
    code: lines`
      pid_m, pid_n = ...                                   # 这个 program 负责的 C tile
      acc = tl.zeros((BM, BN), dtype=tl.float32)
      for k in range(0, K, BK):
          a = tl.load(a_ptrs, mask=..., other=0.0)          # (BM, BK)
          b = tl.load(b_ptrs, mask=..., other=0.0)          # (BK, BN)
          acc += tl.dot(a, b)
          a_ptrs += BK * stride_ak                          # 沿 K 往右移
          b_ptrs += BK * stride_bk                          # 沿 K 往下移
      tl.store(c_ptrs, acc.to(tl.bfloat16), mask=...)
    `,
    ref: '/gpu/triton',
  },
  {
    id: 'triton-group-m',
    deck: 'triton',
    topic: 'matmul',
    q: 'matmul 的 C 有 32 × 32 个 tile，假设同一时刻有 64 个 program 在跑。按行主序给 program 编号，和用 `GROUP_M = 8` 分组编号，这 64 个 program 各要读多少条 A 行条带、B 列条带？为什么分组更快？',
    a: '算 C 的 tile `(m, n)` 要读 A 的第 m 条行条带和 B 的第 n 条列条带。同一时刻在跑的 program 读的条带越少，越容易都留在 L2 里，HBM 流量就越小。\n1. 行主序（`pid_m = pid // 32`，`pid_n = pid % 32`）：64 个 program 正好铺满 C 的 2 行 × 32 列，要读 **2 条 A + 32 条 B = 34 条**。\n2. `GROUP_M = 8`：编号先在 8 行内竖着排，再往右走，64 个 program 是 8 行 × 8 列的方块，要读 **8 条 A + 8 条 B = 16 条**。\n3. 每条条带是 128 × 4096 × 2 B = 1 MiB（BM = 128、K = 4096、bf16），34 MiB 对 16 MiB，后者更容易放进 H100 的 50 MB L2。\n只改了 `pid → (pid_m, pid_n)` 的映射，计算量完全不变，省的是 HBM 读。Triton 官方 matmul 教程里叫 `GROUP_SIZE_M`，见「L2 Cache Optimizations」一节：https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html',
    fig: lines`
      C tiles, numbers = program id (first 64 running)

      row-major            GROUP_M = 8
      row 0:  0 .. 31      rows 0-7, cols 0-7: 0 .. 63
      row 1: 32 .. 63      (8 x 8 square)
      A strips 2, B 32     A strips 8, B 8
    `,
    ref: '/gpu/triton',
  },
  {
    id: 'kernel-parallel-vs-reduce',
    deck: 'triton',
    topic: 'kernel mindset',
    q: '拿到一个 kernel，每个维度先判断什么？',
    a: '判断它是**并行维**还是**归约维**：\n1. 并行维：输出沿这个维度互不依赖 → 直接切开分给不同 program，进 **grid**。\n2. 归约维：多个输入合成一个输出 → 在 program 里**循环 / 累加**。\n3. 例子：softmax / rmsnorm 是行并行、列归约；matmul 是 M、N 并行（2D grid）、K 归约（循环）；attention 是 B、H、Q 并行，KV 长度归约。',
    ref: KM,
  },
  {
    id: 'kernel-elementwise-bound',
    deck: 'triton',
    topic: 'kernel mindset',
    q: '逐元素 kernel（add、relu、cast）为什么一定 memory-bound？能做的优化有哪些？',
    a: '每个输入只读一次、算一次，算术强度极低。能做的只有两件事：**合并访存**，和**与前后 kernel 融合**（省掉中间结果写回显存）。\n和 LLM 的 decode 很像，原因相同：算术强度低。batch = 1 的 decode，每个权重只读一次、用一次（2 字节的 bf16 换 2 次 FLOP，约 1 FLOP/byte），远低于 GPU 的平衡点（H100 约 300 FLOP/byte）。\n区别在于能不能补救：decode 的权重是所有请求共享的，**加大 batch** 就能让一次读取被多个 token 用，算术强度跟着涨；逐元素 kernel 的每个输入本来就只对应一个输出，没有可复用的，只能靠融合、量化少搬字节。',
    ref: KM,
  },
  {
    id: 'kernel-cross-program-reduce',
    deck: 'triton',
    topic: 'kernel mindset',
    q: '一个 program 装不下的 reduce，有哪几种跨 program 的做法？',
    a: '**atomic**（简单，但浮点结果不可复现）、**两遍**（先写局部结果，再起一个 kernel 合并）、**split-K**（matmul 里 K 太长时）。本质都是需要一个全局同步点。\nsplit-K：matmul `C (M×N) = A (M×K) · B (K×N)`，正常是一个 program 算 C 的一个 tile、沿 K 一路循环到底。M、N 小而 K 很长时（比如 decode 的小 batch GEMM），C 的 tile 太少，program 不够把 SM 喂满。\nsplit-K 把 K 也切成 S 段，每段交给一个 program，各自算出这一段的部分和，再合并：`atomic_add` 加进 C，或者写进 `(S, M, N)` 的临时数组、再起一个 kernel 求和。代价是多一次合并，换来 S 倍的并行度。',
    ref: KM,
  },
]

/** 删掉的卡：data 分支和各设备的本地缓存里可能还留着它们的复习记录、批注，页面不显示，测试放行 */
export const retiredIds: string[] = ['torch-read-index', 'torch-write-index', 'bagu-gpu-three-basics', 'bagu-how-many-gpus']

/** 原语卡（torch / triton）+ 八股卡（flashcards-bagu.ts） */
export const cards: Card[] = [...primitiveCards, ...baguCards]

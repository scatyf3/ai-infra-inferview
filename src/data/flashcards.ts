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
    a: '`view` 的本质：新建一个 tensor 对象，指向**同一块 storage**，只换一套 shape 和 stride（必要时还有 offset），一个字节都不拷。所以它只能表达「按某组步长去读同一块内存」读得出来的形状，读不出来就报错。\n例子：attention 输出 `out` 是 `(h, N, dk)`、连续，stride `(N·dk, dk, 1)`。`transpose(0, 1)` 后 shape `(N, h, dk)`，stride `(dk, N·dk, 1)`。\n合成 `(N, h·dk)` 要把 h、dk 两维并成一维。相邻两维能合并的条件是 `stride[h] == shape[dk] × stride[dk]`，这里应该是 `dk`，实际是 `N·dk`：同一个 token 的各个 head 在内存里隔着 N·dk 个元素，不是连着的一段，一个 stride 表达不出来。\n`reshape`：能 view 就 view，不能就先拷一份连续的；`.contiguous()`：手动做这次拷贝。',
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
    a: '想做的：每一行减去这一行的最大值。\n`logits.max(dim=-1).values` 的 shape 是 `(B,)`，被归约的那一维没了。广播先在左边补 1，变成 `(1, B)`，对上的是 `(B, V)` 的**最后一维 V**，不是 B。\n结果：B ≠ V 时报错；B = V 时不报错但算错。看代码里的数：第 0 行应该减 5、第 1 行减 3，实际变成了第 0 列减 5、第 1 列减 3。\n`keepdim=True` 让结果保留成 `(B, 1)`，对上的才是行。归约后还要和原 tensor 运算的，一律加 `keepdim=True`。\n你的 LeetGPU 解法里没出现过这个错，它是 stable-softmax 和 torch 原语页列的常见坑；NumPy 的 `keepdims` 也一样。',
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
    q: '加法 causal mask 用 `triu` 怎么写？写成 `triu(0)` 会怎样？',
    a: '`torch.full((S, S), -inf).triu(1)`：保留主对角线**右上方**为 -inf。`triu(0)` 把对角线也盖掉，第一行全是 -inf，softmax 出 **NaN**。',
    ref: `${TP}#mask`,
  },
  {
    id: 'torch-masked-fill',
    deck: 'torch',
    topic: 'mask',
    q: '`masked_fill` 的语义？`~allowed` 是什么？布尔版 causal mask 怎么写？',
    a: '`x.masked_fill(mask, value)`：mask 为 True 的位置换成 value，其余保留 x 原来的值；mask 广播成 x 的 shape；返回新 tensor（原地版是 `masked_fill_`）。\n`~` 是布尔 tensor 的逐元素取反（True ↔ False）。`allowed` 是「能看」的位置（下三角），`~allowed` 就是「不能看」的位置，把它们填成 -inf。\n放在 softmax **之前**：$e^{-\infty} = 0$，不能看的位置概率就是 0。',
    code: lines`
      allowed = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()   # S = 3 时：
      # [[ True, False, False],      ~allowed: [[False,  True,  True],
      #  [ True,  True, False],                 [False, False,  True],
      #  [ True,  True,  True]]                 [False, False, False]]
      attn = attn.masked_fill(~allowed, float('-inf'))
    `,
    ref: `${TP}#mask`,
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
      x.transpose(0, 1)
      x.transpose(0, 1).stride()
      torch.zeros(2, 3, 4).transpose(-1, -2).shape
    `,
    a: '`transpose(d0, d1)` 对调两个维度的 shape 和 stride，storage 不动：原来的 `x[i, j]` 变成 `y[j, i]`。负数下标从后往前数，`(-1, -2)` 就是最后两维。',
    code: lines`
      >>> x.transpose(0, 1)
      tensor([[0, 3],
              [1, 4],
              [2, 5]])

      >>> x.transpose(0, 1).stride()
      (1, 3)

      >>> torch.zeros(2, 3, 4).transpose(-1, -2).shape
      torch.Size([2, 4, 3])
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
    id: 'torch-read-index',
    deck: 'torch',
    topic: '索引 / 切片 · 读代码',
    q: '读代码：每个表达式输出什么？（索引 / 切片）',
    qcode: lines`
      x = torch.arange(12).view(3, 4)
      x[:, 1:3]
      x[-1]
      x[:, -1]
      x[:, -1:].shape
    `,
    a: '每一维用 `start:stop` 取一段（不含 stop），或用整数取一个位置，负数从后往前数。**整数下标会把那一维去掉**，切片会保留：`x[:, -1]` 是 (3,)，`x[:, -1:]` 是 (3, 1)。都是 view，不拷贝。',
    code: lines`
      >>> x[:, 1:3]
      tensor([[ 1,  2],
              [ 5,  6],
              [ 9, 10]])

      >>> x[-1]
      tensor([ 8,  9, 10, 11])

      >>> x[:, -1]
      tensor([ 3,  7, 11])

      >>> x[:, -1:].shape
      torch.Size([3, 1])
    `,
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-write-index',
    deck: 'torch',
    topic: '索引 / 切片 · 写代码',
    q: '写代码：decoder 的 hidden states `h` 是 `(B, S, D)`（B batch，S 序列长度，D hidden 维度）。生成下一个 token 只要每个序列**最后一个位置**的向量，得到 `(B, D)`。',
    a: '`h[:, -1]`（等于 `h[:, -1, :]`）。想保留序列那一维就写 `h[:, -1:]`，得到 `(B, 1, D)`。',
    ref: `${TP}#tensor-itself`,
  },
  {
    id: 'torch-read-expand',
    deck: 'torch',
    topic: 'expand · 读代码',
    q: '读代码：每个表达式输出什么？（expand）',
    qcode: lines`
      x = torch.tensor([[1], [2]])
      x.expand(2, 3)
      x.expand(-1, 3).stride()
    `,
    a: '`expand(*shape)` 把长度为 1 的维「拉长」到给定长度，`-1` 表示这一维不变。不拷贝：被拉长的那一维 stride 是 0，每一列读到的都是同一个元素。只能拉长长度为 1 的维。',
    code: lines`
      >>> x.expand(2, 3)
      tensor([[1, 1, 1],
              [2, 2, 2]])

      >>> x.expand(-1, 3).stride()
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
    q: '写代码：位置编码表 `pe` 的 shape `(S, D)`（S 序列长度，D hidden 维度），要给 batch 里 B 个样本各一份**能独立改写**的拷贝，得到 `(B, S, D)`。',
    a: '`pe.repeat(B, 1, 1)`：`pe` 先当成 `(1, S, D)`，第 0 维铺 B 份。只读不写的话用 `pe.expand(B, S, D)`，不拷贝。',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-read-repeat-interleave',
    deck: 'torch',
    topic: 'repeat_interleave · 读代码',
    q: '读代码：每个表达式输出什么？（repeat_interleave）',
    qcode: lines`
      x = torch.tensor([1, 2])
      x.repeat_interleave(3)
      torch.tensor([[1, 2], [3, 4]]).repeat_interleave(2, dim=0)
    `,
    a: '`repeat_interleave(n, dim)` 把**每个元素**（或沿 dim 的每一片）原地连着重复 n 次：[1, 2] 变成 [1, 1, 1, 2, 2, 2]。对比 `repeat` 是整体平铺：[1, 2, 1, 2, 1, 2]。不传 dim 时先展平。真的拷贝。',
    code: lines`
      >>> x.repeat_interleave(3)
      tensor([1, 1, 1, 2, 2, 2])

      >>> torch.tensor([[1, 2], [3, 4]]).repeat_interleave(2, dim=0)
      tensor([[1, 2],
              [1, 2],
              [3, 4],
              [3, 4]])
    `,
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-write-repeat-interleave',
    deck: 'torch',
    topic: 'repeat_interleave · 写代码',
    q: '写代码：GQA：k 的 shape `(B, H_kv, S, dk)`，每个 KV head 给连续的 g 个 Q head 用：Q head i 用 KV head `i // g`，H = H_kv · g。想展开成 `(B, H, S, dk)` 和 q 一一对齐。',
    a: '`k.repeat_interleave(g, dim=1)`：KV head 的顺序变成 0, 0, …, 1, 1, …，正好是 `i // g`。用 `k.repeat(1, g, 1, 1)` 会变成 0, 1, …, 0, 1, …，对应的是 `i % H_kv`，**错了还不报错**。',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-read-cat',
    deck: 'torch',
    topic: 'cat · 读代码',
    q: '读代码：每个表达式输出什么？（cat）',
    qcode: lines`
      a = torch.tensor([[1, 2]])
      b = torch.tensor([[3, 4]])
      torch.cat([a, b])
      torch.cat([a, b], dim=1)
    `,
    a: '`torch.cat(tensors, dim=0)` 沿**已有的**第 dim 维首尾相接，其他维必须一样长，维度数不变。默认 dim = 0。',
    code: lines`
      >>> torch.cat([a, b])
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
    a: '`k_cache = torch.cat([k_cache, k_new], dim=2)`，得到 `(B, H, T + 1, dk)`。每步都要把整个 cache 拷一遍，所以推理框架用预分配 + 原地写，不用 cat。',
    ref: `${TP}#只改元数据-vs-会拷贝`,
  },
  {
    id: 'torch-read-stack',
    deck: 'torch',
    topic: 'stack · 读代码',
    q: '读代码：每个表达式输出什么？（stack）',
    qcode: lines`
      a = torch.tensor([1, 2])
      b = torch.tensor([3, 4])
      torch.stack([a, b])
      torch.stack([a, b], dim=1)
    `,
    a: '`torch.stack(tensors, dim)` 先给每个 tensor 在第 dim 位**新插一维**再拼，所有 tensor 的 shape 必须完全一样，结果多一维。对比 `cat` 是沿已有的维拼，维度数不变。',
    code: lines`
      >>> torch.stack([a, b])
      tensor([[1, 2],
              [3, 4]])

      >>> torch.stack([a, b], dim=1)
      tensor([[1, 3],
              [2, 4]])
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
    a: '`x - x.max(dim=-1, keepdim=True).values`：减数是 `(B, 1)`，按行广播。漏了 keepdim 就是 `(B,)`，对上的是 V 那一维。',
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
    topic: 'triu · 写代码',
    q: '写代码：加法 causal mask：`(S, S)`，query i 只能看 key j ≤ i；看不到的位置是 -inf、能看到的是 0，加到 score 上。要和 score `x` 在同一个 device。',
    a: '`torch.full((S, S), float(\'-inf\'), device=x.device).triu(1)`：右上方（j > i）留 -inf，其余变 0。写成 `triu()` 会把对角线也盖成 -inf，第一行全是 -inf，softmax 出 NaN。',
    ref: `${TP}#mask`,
  },
  {
    id: 'torch-read-tril',
    deck: 'torch',
    topic: 'tril · 读代码',
    q: '读代码：每个表达式输出什么？（tril）',
    qcode: lines`
      torch.ones(3, 3, dtype=torch.bool).tril()
      torch.arange(1, 10).view(3, 3).tril(-1)
    `,
    a: '`tril(k)` 保留第 k 条对角线及它**左下方**的元素，其余置 0（bool 是 False）。k = -1 不含主对角线。',
    code: lines`
      >>> torch.ones(3, 3, dtype=torch.bool).tril()
      tensor([[ True, False, False],
              [ True,  True, False],
              [ True,  True,  True]])

      >>> torch.arange(1, 10).view(3, 3).tril(-1)
      tensor([[0, 0, 0],
              [4, 0, 0],
              [7, 8, 0]])
    `,
    ref: `${TP}#mask`,
  },
  {
    id: 'torch-write-tril',
    deck: 'torch',
    topic: 'tril · 写代码',
    q: '写代码：布尔 causal mask：`(S, S)`，`True` 表示 query i 能看 key j（j ≤ i）。要和 score `x` 在同一个 device。',
    a: '`allowed = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()`。之后用 `x.masked_fill(~allowed, float(\'-inf\'))` 把看不到的位置盖掉。',
    ref: `${TP}#mask`,
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
      ~m
      x.masked_fill(~m, 0.)
    `,
    a: '`x.masked_fill(mask, value)`：mask 为 True 的位置换成 value，其余保留 x 的值；mask 会广播成 x 的 shape；返回新 tensor。`~` 是布尔取反（True ↔ False），所以 `masked_fill(~m, v)` 填的是 m 为 False 的位置。',
    code: lines`
      >>> x.masked_fill(m, 0.)
      tensor([[1., 0.],
              [3., 4.]])

      >>> ~m
      tensor([[ True, False],
              [ True,  True]])

      >>> x.masked_fill(~m, 0.)
      tensor([[0., 2.],
              [0., 0.]])
    `,
    ref: `${TP}#mask`,
  },
  {
    id: 'torch-write-masked-fill',
    deck: 'torch',
    topic: 'masked_fill · 写代码',
    q: '写代码：padding mask：`pad` 是 `(B, S)`，`True` 表示这个位置是补出来的 padding。score `s` 是 `(B, H, S, S)`（最后一维是 key），任何 query 都不能看 padding 的 key。',
    a: '`s.masked_fill(pad[:, None, None, :], float(\'-inf\'))`：pad 变成 `(B, 1, 1, S)`，广播到每个 head、每个 query，只盖 key 那一维。',
    ref: `${TP}#mask`,
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
    q: '写代码：online softmax 的运行最大值 `m`：shape `(B, H)`，初值全是 -inf，和 `q` 同 dtype、同 device。',
    a: '`m = torch.full((B, H), float(\'-inf\'), dtype=q.dtype, device=q.device)`。不传 device，和 GPU 上的 tensor 一运算就报 device 不一致。',
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
      y.data_ptr() == x.data_ptr()
      z = torch.arange(6)
      z.contiguous() is z
    `,
    a: '`contiguous()` 返回按当前 shape 行优先连续存放的 tensor：不连续就拷一份（新内存，stride 变成 (2, 1)）；已经连续就直接返回自己，不拷贝。',
    code: lines`
      >>> x.is_contiguous()
      False

      >>> y.stride()
      (2, 1)

      >>> y.data_ptr() == x.data_ptr()
      False

      >>> z.contiguous() is z
      True
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
    q: 'grid 怎么写？块数为什么要向上取整？',
    a: '`grid = (triton.cdiv(n, BLOCK),)`。向下取整会漏掉最后不满一块的元素（它们没有 program 负责，输出是垃圾值）；向上取整多开一块，越界部分用 mask 挡掉。`cdiv(a, b) = (a + b - 1) // b`。',
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
    q: '二维 tile 的指针矩阵怎么拼？',
    a: '两个下标向量广播：`(BM, 1)` 加 `(1, BN)` 得 `(BM, BN)`，规则和 torch 一样。Triton 不认 shape，只认指针，地址靠 stride 自己算。',
    code: 'ptrs = base + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n',
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
    q: '`num_warps` 怎么选？',
    a: '每个线程处理 `BLOCK / (32 × num_warps)` 个元素。块小时 4 个 warp 够；块大时加 warp，降低每线程的寄存器压力。Triton 教程按 BLOCK 用启发式（2048 以上 8，4096 以上 16），生产上交给 autotune。',
    ref: SM,
  },
  {
    id: 'triton-no-grid-sync',
    deck: 'triton',
    topic: '跨 program',
    q: '为什么「要等全局结果」的地方都得切一个新 kernel？',
    a: 'Triton 里不同 program 之间**没有全局同步**。唯一能保证「上一步所有 program 都写完了」的就是 kernel 边界（或者 atomic / 计数器）。\nCUDA 呢：\n1. block 内：`__syncthreads()`，同一个 block 的线程互相等。\n2. 跨 block：普通 launch 也没有。cooperative groups 有 `cg::this_grid().sync()`，但要用 `cudaLaunchCooperativeKernel` 启动，而且所有 block 必须**同时驻留**在 GPU 上（grid 不能超过 SM 数 × 每个 SM 能放的 block 数），否则已经在跑的 block 等不到还没上去的 block，会死锁。所以实际也常拆 kernel 或用 atomic。\n3. Triton 的 program 内部：同步由编译器插，不用自己写（`tl.debug_barrier()` 只用于调试）。',
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
    q: '已知每块的 `(m_b, d_b)`，`d_b = Σ exp(x - m_b)`，全局 M、D 怎么得到？',
    a: '每块先只看自己：\n`m_b = max_{i∈b} x_i`\n`d_b = Σ_{i∈b} exp(x_i − m_b)`\n合并：\n`M = max_b m_b`\n`D = Σ_b d_b · exp(m_b − M)`\n每块的和是相对 m_b 算的，乘 `exp(m_b − M)` 换算成相对 M。因为 m_b ≤ M，修正因子 ≤ 1，不会溢出。\n流式版（一块块扫，FlashAttention 用的就是这个）：\n`m_new = max(m, m_b)`\n`d = d · exp(m − m_new) + d_b · exp(m_b − m_new)`\n初值 `(m, d) = (−inf, 0)`。这个合并满足结合律，所以既能并行两两合并，也能串行一块块扫。',
    code: "# 第 1 趟：每块的局部量\nm_b = tl.max(x, axis=0)\nd_b = tl.sum(tl.exp(x - m_b), axis=0)\n\n# 第 2 趟：合并（ms、ds 是所有块的 m_b、d_b，补齐的位置分别填 -inf、0）\nM = tl.max(ms, axis=0)\nD = tl.sum(ds * tl.exp(ms - M), axis=0)\n\n# 流式版：在一个 program 里一块块扫\nm = tl.full([], -float('inf'), tl.float32)   # 循环里会重新赋值，初值类型要和循环里一致\nd = tl.full([], 0.0, tl.float32)\nfor start in range(0, N, BLOCK):\n    x = tl.load(x_ptr + start + offs, mask=start + offs < N, other=-float('inf'))\n    m_new = tl.maximum(m, tl.max(x, axis=0))\n    d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)\n    m = m_new",
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
    a: '同一地址的 atomic 在 L2 上**串行**。逐元素 atomic（50 万次打一个地址）很慢；先块内 `tl.sum` 归约、每块一次，只有 NB 次，可以忽略。这时真正多出来的是初始化带来的额外 launch。',
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
    q: 'Triton matmul 里一个 program 负责什么？K 维怎么处理？',
    a: '负责 C 的一个 `(BM, BN)` tile。沿 K 循环，每步读 A 的 `(BM, BK)` 和 B 的 `(BK, BN)`，`acc += tl.dot(a, b)`；`acc` 是寄存器里的 fp32 累加器，最后写回一次。M、N 进 grid，K 进循环。',
    ref: '/gpu/triton',
  },
  {
    id: 'triton-group-m',
    deck: 'triton',
    topic: 'matmul',
    q: 'matmul 的 `GROUP_M` 改了什么？为什么能变快？',
    a: '只改 `pid → (pid_m, pid_n)` 的映射：同一波并发的 program 挤在 GROUP_M 行里，而不是排成一整行。这一波要读的 A 行条带 + B 列条带更少，更容易命中 L2；计算量不变，HBM 流量变小。',
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

/** 原语卡（torch / triton）+ 八股卡（flashcards-bagu.ts） */
export const cards: Card[] = [...primitiveCards, ...baguCards]

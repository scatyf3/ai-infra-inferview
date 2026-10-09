---
title: C++：RAII、移动语义、智能指针、模板、内存布局与 Python 绑定
status: draft
tags: [cpp, raii, pybind11, abi]
difficulty: 2
order: 2
related: [/basics/rust, /basics/python, /framework/pytorch-internals, /gpu/gpu-architecture, /handson/cuda-tiled-matmul]
stack: []
---

# C++：RAII、移动语义、智能指针、模板、内存布局与 Python 绑定

> kernel / framework 岗常问；PyTorch 的 ATen / c10、vLLM 的 `csrc/`、FlashAttention、TensorRT-LLM 都是 C++ / CUDA

## 一句话结论

infra 岗的 C++ 考的是「资源归谁管、什么时候释放、调用开销在哪、二进制能不能链上」：
1. RAII 把资源（显存、stream、锁）的生命周期绑到对象上，异常路径也不会漏；
2. 移动语义让大对象只搬指针不拷数据，智能指针把所有权写进类型；
3. 模板把 dtype、tile 大小变成编译期常量，换来展开和特化，代价是编译时间和二进制体积；
4. 对齐决定能不能用向量化访存；
5. pybind11 / `TORCH_LIBRARY` 把 C++ 暴露给 Python，最常见的坑是 ABI 不匹配。

## 推导

### 0. 几个前置定义

- **栈对象 vs 堆对象**：局部变量在栈上，离开作用域自动析构；`new` / `malloc` 出来的在堆上，要有人负责 `delete` / `free`。
- **左值 / 右值**：有名字、能取地址的是左值（`x`）；临时值是右值（`f()` 的返回值、`x + 1`）。`T&&` 是右值引用，只能绑定到右值。
- **所有权**：「谁负责释放」。C++ 语言本身不检查所有权，靠约定和类型（`unique_ptr` 等）表达；Rust 把它写进了编译期检查，对照见 [Rust：所有权](/basics/rust#_1-所有权与-move-比-c-move-更严格)。

### 1. RAII：资源跟着对象走

**定义**：Resource Acquisition Is Initialization，构造函数获取资源，析构函数释放资源（[C++ Core Guidelines：R.1](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#rr-raii)）。因为栈对象在作用域结束时**一定**析构（包括抛异常时的栈展开），资源就不会泄漏。

一个持有显存的 RAII 类，顺便演示「禁拷贝、可移动」（下一节解释移动）：

```cpp
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); \
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while (0)

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    CUDA_CHECK(cudaMalloc(&ptr_, bytes));        // 构造 = 获取
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }  // 析构 = 释放（析构里不抛异常）

  DeviceBuffer(const DeviceBuffer&) = delete;             // 禁止拷贝：两个对象 free 同一指针 = double free
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept                 // 移动：偷指针，把源置空
      : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = std::exchange(o.ptr_, nullptr);
      bytes_ = std::exchange(o.bytes_, 0);
    }
    return *this;
  }

  void* data() const { return ptr_; }
  size_t size() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

void forward(bool fail) {
  DeviceBuffer workspace(64 << 20);   // 64 MB 临时显存
  if (fail) throw std::runtime_error("shape mismatch");
}   // 正常返回或抛异常，workspace 都会析构

int main() {
  try { forward(true); } catch (const std::exception&) {}
  std::vector<DeviceBuffer> pool;
  for (int i = 0; i < 4; ++i) pool.emplace_back(1 << 20);   // 扩容时走 noexcept 移动，不拷贝
  DeviceBuffer a(1024);
  DeviceBuffer b = std::move(a);      // a.data() 现在是 nullptr
}
```

（把 `cudaMalloc` 换成 `malloc` 的同构版本已在 clang 17 `-std=c++17` 下编译运行通过。）

**Rule of 0 / 3 / 5**（[cppreference](https://en.cppreference.com/w/cpp/language/rule_of_three)）：类如果自己管理资源，就要把析构、拷贝构造、拷贝赋值、移动构造、移动赋值这五个一起考虑；更好的做法是 rule of 0，成员全用已经是 RAII 的类型（`unique_ptr`、`std::vector`），一个都不用手写。

infra 里的 RAII 实例：`std::lock_guard`（锁）、`c10::cuda::CUDAGuard`（切换当前 device，析构时切回）、`c10::cuda::CUDAStreamGuard`、PyTorch 的 `at::AutoDispatchBelowAutograd`、pybind11 的 `py::gil_scoped_release`（第 6 节）。

### 2. 移动语义

**问题**：函数返回一个 `std::vector<float>`，或者 vector 扩容时搬旧元素，如果每次都深拷贝，就是 O(n) 的内存拷贝。

**做法**：给类加一个以 `T&&` 为参数的移动构造函数，实现为「偷走源对象的指针，把源置空」，O(1)。

- `std::move(x)` **本身不移动任何东西**，只是把 `x` 转成右值引用，让重载决议选中移动构造。
- 被移走的标准库对象处于「有效但未指定」的状态：可以析构、可以重新赋值，但不要假设它的内容。
- **`noexcept` 很关键**：`std::vector` 扩容时用 `std::move_if_noexcept`，只有移动构造声明了 `noexcept` 才会移动，否则为了强异常安全退回拷贝（[cppreference：move_if_noexcept](https://en.cppreference.com/w/cpp/utility/move_if_noexcept)）。上面的 `DeviceBuffer` 禁了拷贝，如果移动不是 `noexcept`，vector 仍会用移动（因为别无选择），但对可拷贝的大对象，漏写 `noexcept` 会让每次扩容都深拷贝。
- 返回局部变量直接 `return x;`，不要写 `return std::move(x);`：前者允许 NRVO（直接在调用方构造，连移动都省了）。

### 3. 智能指针

| | `unique_ptr<T>` | `shared_ptr<T>` | `weak_ptr<T>` | 裸指针 `T*` / 引用 |
| --- | --- | --- | --- | --- |
| 语义 | 独占所有权 | 共享所有权，计数归零释放 | 不拥有，观察 `shared_ptr` 管的对象 | 借用，不拥有 |
| 大小（64 位） | 8 字节（默认 deleter） | 16 字节（对象指针 + 控制块指针） | 16 字节 | 8 字节 |
| 拷贝开销 | 不可拷贝，只能移动 | 原子加减引用计数 | 原子加减弱计数 | 无 |

**原子计数的代价**（实测，Apple M 系列，clang `-O2`）：单线程拷贝 + 析构一个 `shared_ptr` 约 5 ns；8 个线程同时拷贝同一个 `shared_ptr`，计数所在的 cache line 在核之间来回传，每次升到几百 ns。所以热路径里传参用 `const shared_ptr<T>&` 或直接传 `T&` / `T*`，只在真要延长生命周期时才拷贝。

- `make_shared<T>()` 把对象和控制块一次分配在一起，比 `shared_ptr<T>(new T)` 少一次堆分配。
- 循环引用（A 持有 B 的 `shared_ptr`，B 也持有 A 的）永远不会归零，其中一方改用 `weak_ptr`。
- **PyTorch 用的是侵入式计数**：`TensorImpl` 继承 `c10::intrusive_ptr_target`，计数存在对象内部（[TensorImpl.h](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h)、[intrusive_ptr.h](https://github.com/pytorch/pytorch/blob/main/c10/util/intrusive_ptr.h)）。`at::Tensor` 本身就是一个 `intrusive_ptr<TensorImpl>`，拷贝 `Tensor` 只是加一次计数，不拷数据。

### 4. 模板基础：把 dtype 和 tile 大小变成编译期常量

**定义**：函数模板 / 类模板是「生成代码的配方」，每用一组新的模板参数，编译器就**实例化**出一份独立的函数（单态化）。模板参数可以是类型，也可以是整数常量。

为什么 kernel 几乎都写成模板：tile 大小、展开因子、dtype 是编译期常量时，编译器能完全展开循环、把数组放进寄存器、选对应的向量化指令。运行时的 dtype 再用一个 `switch` 映射到某个实例，ATen 的 `AT_DISPATCH_FLOATING_TYPES` 等宏做的就是这件事（[ATen/Dispatch.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h)）。

```cpp
#include <cstdint>
#include <stdexcept>

enum class DType { F32, F64 };

template <typename T, int kUnroll>
void scale_kernel(T* x, int64_t n, T s) {        // CPU 版示意；CUDA 里就是 __global__ 函数模板
  int64_t i = 0;
  for (; i + kUnroll <= n; i += kUnroll) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) x[i + u] *= s;   // kUnroll 是编译期常量，可完全展开
  }
  for (; i < n; ++i) x[i] *= s;                         // 尾部
}

// 运行时 dtype → 编译期类型
void scale(void* x, int64_t n, double s, DType dt) {
  switch (dt) {
    case DType::F32: scale_kernel<float, 4>(static_cast<float*>(x), n, float(s)); break;
    case DType::F64: scale_kernel<double, 2>(static_cast<double*>(x), n, s); break;
    default: throw std::invalid_argument("unsupported dtype");
  }
}
```

**代价**：实例数 = 各参数取值个数的乘积。比如 4 种 dtype × 3 种 head_dim × 2 种 causal × 2 种 paged，就是 48 份 kernel，这就是 FlashAttention、vLLM 这类项目编译动辄几十分钟、wheel 几百 MB 的原因。常见缓解：只实例化真正用到的组合，拆成多个 `.cu` 并行编译。

### 5. 虚函数与 vtable

每个含虚函数的类有一张 vtable（函数指针数组），每个对象多存一个 vptr（8 字节）。虚调用 = 读 vptr → 读表项 → 间接跳转，多两次访存，且编译器通常无法内联。在每秒调用百万次的算子分发路径上这很可观，PyTorch 的 dispatcher 用的是按 dispatch key 索引的函数指针表（`OperatorEntry` 里的 kernel 表），而不是对每个 Tensor 做虚调用。GPU kernel 里基本不用虚函数。

### 6. 内存布局与对齐

**对齐规则**：类型 `T` 的地址必须是 `alignof(T)` 的倍数；结构体成员按声明顺序排，编译器在中间插 padding 满足对齐，结构体总大小是其最大成员对齐的倍数。

```cpp
#include <cstddef>
#include <cstdint>

struct Bad  { char flag; double scale; int32_t id; };   // 1 + 7(pad) + 8 + 4 + 4(pad) = 24
struct Good { double scale; int32_t id; char flag; };   // 8 + 4 + 1 + 3(pad) = 16
struct alignas(16) Vec4 { float x, y, z, w; };          // 对应 CUDA 的 float4

static_assert(sizeof(Bad) == 24 && alignof(Bad) == 8);
static_assert(offsetof(Bad, scale) == 8);
static_assert(sizeof(Good) == 16);
static_assert(sizeof(Vec4) == 16 && alignof(Vec4) == 16);
```

为什么和 GPU 有关：

1. **向量化访存**：一个线程用 `float4` 一次读 16 字节，要求地址 16 字节对齐。`cudaMalloc` 返回的地址至少 256 字节对齐（[CUDA Programming Guide 12.6：Device Memory Accesses](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#device-memory-accesses)），但 tensor 切片之后的 `data_ptr()` 可能落在任意元素边界，所以 kernel 里要检查 `ptr % 16 == 0` 再走向量化分支。
2. **合并访存**：一个 warp 的 32 个线程访问连续且对齐的地址时，硬件合并成最少的内存事务。详见 [GPU 架构](/gpu/gpu-architecture)。
3. **约定**：本站默认 row-major，`[M, N]` 矩阵元素 `(i, j)` 的偏移是 `i * N + j`；PyTorch tensor 用 `stride` 表达任意布局，`.contiguous()` 才保证是这个公式。
4. **主机和设备共享的结构体**（比如传给 kernel 的参数 struct）两边必须布局一致，用 `static_assert(sizeof(...))` 锁住。

### 7. 把 C++ 暴露给 Python：pybind11 与 `TORCH_LIBRARY`

**pybind11**：用模板在编译期生成 Python ↔ C++ 的类型转换代码。长计算要释放 GIL，否则同进程的其他 Python 线程（包括 asyncio 事件循环）全被卡住（[pybind11：GIL](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)，GIL 本身见 [Python](/basics/python#_1-gil-是什么-何时释放)）：

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
namespace py = pybind11;

std::vector<int> count_words(const std::vector<std::string>& texts) {
  std::vector<int> out;
  for (const auto& t : texts) {
    int n = 0; bool in = false;
    for (char c : t) { bool sp = (c == ' '); n += (!sp && !in); in = !sp; }
    out.push_back(n);
  }
  return out;
}

PYBIND11_MODULE(fastcount, m) {
  // 参数先在持有 GIL 时从 list[str] 转成 std::vector<std::string>（这一步是拷贝），
  // 然后 call_guard 在函数体执行期间释放 GIL
  m.def("count_words", &count_words, py::call_guard<py::gil_scoped_release>());
}
```

释放 GIL 期间不能碰任何 `py::object`。Rust 的 PyO3 是同一套思路（`py.detach`），对照见 [Rust：PyO3](/basics/rust#_10-pyo3-给-python-写扩展并释放-gil)。传 numpy 数组用 `py::array_t<float>` 走 buffer protocol，不拷数据（[pybind11：NumPy](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)）。

**PyTorch 自定义算子**：推荐用 `TORCH_LIBRARY` 注册，而不是直接用 pybind11 暴露函数。注册后算子出现在 `torch.ops.<ns>.<name>`，能被 `torch.compile` 和 dispatcher 识别（[Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)）：

```cpp
#include <torch/library.h>
#include <ATen/ATen.h>

at::Tensor muladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  return a * b + c;
}

TORCH_LIBRARY(myops, m) {            // 第一步：定义 schema
  m.def("muladd(Tensor a, Tensor b, float c) -> Tensor");
}
TORCH_LIBRARY_IMPL(myops, CPU, m) {  // 第二步：给某个 backend 注册实现（CUDA 同理）
  m.impl("muladd", &muladd_cpu);
}
```

schema 里的 `float` 对应 C++ 的 `double`。vLLM 的 `csrc/` 下的自定义 kernel（paged attention、量化 GEMM 等）就是这样注册成 `torch.ops._C.*` 的。

### 8. ABI 坑

**定义**：API 是源码层面的接口；ABI（Application Binary Interface）是二进制层面的约定：函数符号怎么命名（name mangling）、参数怎么传、结构体怎么布局、vtable 怎么排。Linux 上 GCC / Clang 遵循 Itanium C++ ABI（[规范](https://itanium-cxx-abi.github.io/cxx-abi/abi.html)）。两个 `.so` 能链在一起，靠的是 ABI 一致，而不是源码能编译。

infra 里最常见的三类报错：

1. **`_GLIBCXX_USE_CXX11_ABI` 不一致**：GCC 5 起 `std::string` / `std::list` 有新旧两套实现，新版符号在 `std::__cxx11` 命名空间里，由这个宏选择（[libstdc++：Dual ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)）。你的扩展和 libtorch 用了不同的值，就会出现带 `__cxx11` 的 `undefined symbol`。用 `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` 看 torch 用的是哪个，扩展用同一个值编译。`torch.utils.cpp_extension` 会自动帮你对齐。
2. **torch 版本不一致**：libtorch 的 C++ API 不保证跨版本 ABI 稳定，升级 torch 后旧的扩展 `.so`（包括 pip 装的 flash-attn、vLLM 的 `_C.so`）会报 `undefined symbol`，必须对着新 torch 重新编译。这就是这些包的 wheel 名里要写 torch 版本的原因。PyTorch 新增了 ABI 稳定的 `STABLE_TORCH_LIBRARY` + `Py_LIMITED_API`，一份 wheel 可以跨 torch / Python 版本（见上面的 custom ops 教程）。
3. **CUDA 版本不一致**：扩展用 CUDA 12.4 编译，torch 是 cu121 的，可能出现 `undefined symbol` 或运行时找不到 `libcudart.so.12.x`。`torch.version.cuda` 和 `nvcc --version` 要对上大版本。

ABI 边界上还要遵守「谁分配谁释放」：A 库 `new` 出来的对象交给 B 库 `delete`，两边若链的是不同的 C++ 运行时或 allocator，就会崩。跨库接口传 `std::string`、STL 容器也同样受上面第 1 点影响，所以真正稳定的跨库接口一般是 C 函数 + POD 结构体（`extern "C"`）。

## 面试追问

::: details Q：shared_ptr 的引用计数为什么是原子的，开销有多大？
因为多个线程可能同时拷贝 / 析构指向同一对象的 `shared_ptr`，计数必须用原子增减。无竞争时一次拷贝 + 析构是几 ns；多个核同时操作同一个控制块时，cache line 在核间来回迁移，单次能到几百 ns（上面的实测）。所以函数参数按 `const shared_ptr&` 或 `T&` 传，只在真正需要延长生命周期时拷贝。注意：原子的只是计数，被指对象本身的读写并不是线程安全的。
:::

::: details Q：std::move 之后的对象还能用吗？
`std::move` 只是一个到右值引用的转换，真正的移动发生在移动构造 / 移动赋值里。标准库类型被移走后处于「有效但未指定」状态：可以析构、可以重新赋值、可以调用没有前置条件的函数（如 `clear()`、`size()`），但不能假设它的内容。自定义类型由你决定，惯例是像上面 `DeviceBuffer` 那样把源置空。Rust 在编译期禁止使用被移走的变量，C++ 不禁止，误用是运行时 bug。
:::

::: details Q：为什么移动构造要写 noexcept？
`std::vector` 扩容要把旧元素搬到新内存。如果搬到一半抛异常，用拷贝的话旧数据还在，可以回滚；用移动的话旧数据已经被偷走，无法回滚。为了保持强异常安全，vector 通过 `std::move_if_noexcept` 只在移动构造是 `noexcept`（或类型不可拷贝）时才移动。漏写 `noexcept` 的可拷贝大对象，每次扩容都会被深拷贝。
:::

::: details Q：import 自己编译的 CUDA 扩展报 undefined symbol，怎么排查？
1. `c++filt` 解码报错里的符号名：带 `__cxx11` 的多半是 `_GLIBCXX_USE_CXX11_ABI` 不一致；带 `c10::` / `at::` 的多半是 torch 版本不一致；带 `cuda` / `cublas` 的看 CUDA 版本。
2. `nm -D --defined-only libtorch_cpu.so | grep <符号>` 看 torch 是否导出了这个符号。
3. `ldd your_ext.so` 看实际加载的是哪个 libtorch / libcudart。
4. 确认用当前环境的 torch 重新编译（`pip install --no-build-isolation` 避免在隔离环境里拉一个不同版本的 torch 来编译）。
:::

## 手撕

### 简化版 unique_ptr

要点：禁拷贝、可移动、析构释放、`release` 交出所有权、`reset` 先换后删（自赋值安全），大小和裸指针一样。

```cpp
#include <cstdio>
#include <utility>

template <typename T>
class UniquePtr {
 public:
  UniquePtr() = default;
  explicit UniquePtr(T* p) noexcept : p_(p) {}
  ~UniquePtr() { delete p_; }

  UniquePtr(const UniquePtr&) = delete;
  UniquePtr& operator=(const UniquePtr&) = delete;

  UniquePtr(UniquePtr&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
  UniquePtr& operator=(UniquePtr&& o) noexcept {
    if (this != &o) reset(o.release());
    return *this;
  }

  T* get() const noexcept { return p_; }
  T& operator*() const { return *p_; }
  T* operator->() const noexcept { return p_; }
  explicit operator bool() const noexcept { return p_ != nullptr; }

  T* release() noexcept { return std::exchange(p_, nullptr); }  // 交出所有权，不释放
  void reset(T* p = nullptr) noexcept {                          // 先换再删
    T* old = std::exchange(p_, p);
    delete old;
  }

 private:
  T* p_ = nullptr;
};

template <typename T, typename... Args>
UniquePtr<T> make_unique_ptr(Args&&... args) {
  return UniquePtr<T>(new T(std::forward<Args>(args)...));       // 完美转发构造参数
}

struct Req {
  explicit Req(int i) : id(i) {}
  ~Req() { std::printf("free req %d\n", id); }
  int id;
};

int main() {
  auto a = make_unique_ptr<Req>(1);
  UniquePtr<Req> b = std::move(a);
  std::printf("a empty=%d b->id=%d\n", !a, b->id);   // a empty=1 b->id=1
  b.reset(new Req(2));                                // 打印 free req 1
  static_assert(sizeof(UniquePtr<Req>) == sizeof(Req*));
}                                                     // 打印 free req 2
```

追问常见扩展：支持数组（`UniquePtr<T[]>` 特化，析构用 `delete[]`）、自定义 deleter（比如 `cudaFree`，作为模板参数时用空基类优化保持 8 字节）。

## 参考

- [C++ Core Guidelines：资源管理](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#s-resource)
- [cppreference：std::unique_ptr](https://en.cppreference.com/w/cpp/memory/unique_ptr)
- [cppreference：Rule of three/five/zero](https://en.cppreference.com/w/cpp/language/rule_of_three)
- [pybind11：Miscellaneous（GIL）](https://pybind11.readthedocs.io/en/stable/advanced/misc.html)
- [PyTorch：Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [libstdc++：Dual ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)
- [Itanium C++ ABI](https://itanium-cxx-abi.github.io/cxx-abi/abi.html)

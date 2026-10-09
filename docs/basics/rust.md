---
title: Rust：所有权、借用、trait、并发与 PyO3
status: draft
tags: [rust, ownership, concurrency, pyo3]
difficulty: 2
order: 2.5
related: [/basics/cpp, /basics/python]
stack: []
---

# Rust：所有权、借用、trait、并发与 PyO3

> 推理服务的 tokenizer、权重格式、router / gateway 越来越多用 Rust 写；按 C++ 的概念对照着学最快

## 一句话结论

Rust 把 C++ 里靠约定的规则写进了类型系统并在编译期检查：每个值只有一个 owner（move 之后原变量不能再用），引用要么多个只读 `&T`、要么一个可写 `&mut T`，所以悬垂指针和数据竞争在 safe Rust 里编译不过。错误用 `Result` + `?` 返回而不抛异常，多态默认走单态化（像模板），需要时用 `dyn Trait`（像虚函数）。AI infra 里用它写 CPU 侧的热路径，再用 PyO3 暴露给 Python，并在计算期间释放 GIL。

## 推导

### 0. 为什么 AI infra 里会碰到 Rust

| 项目 | Rust 做什么 | 说明 |
| --- | --- | --- |
| [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | 整个分词核心 | Rust 实现，Python 包在 `bindings/python` 里用 PyO3 包一层；`encode_batch` 用 Rayon 多线程 |
| [huggingface/safetensors](https://github.com/huggingface/safetensors) | 权重文件读写 | 替代 pickle，避免反序列化执行任意代码，支持零拷贝 / mmap 加载 |
| [TGI](https://github.com/huggingface/text-generation-inference) | router（webserver） | Rust 写的 HTTP 服务接请求、排队、组 batch，再通过 gRPC 调 Python model server（[架构文档](https://huggingface.co/docs/text-generation-inference/architecture)） |
| [SGLang Model Gateway](https://docs.sglang.io/advanced_features/sgl_model_gateway.html) | 多实例路由 / 网关 | 原 sgl-router（文档里服务名仍叫 `sgl-router`），cargo 构建；负责 worker 管理、负载均衡、重试、限流，gRPC 模式下 tokenizer / reasoning parser / tool parser 都是 Rust 实现 |
| [huggingface/candle](https://github.com/huggingface/candle) | 推理框架 | 纯 Rust 的轻量 ML 框架，API 类似 PyTorch，支持 CPU / CUDA / Metal / WASM，适合 serverless 这类要求启动快、二进制小的场景 |

共同点：都是 **CPU 密集或高并发 IO、又在 Python 进程旁边** 的组件。它们要的是 C++ 的性能，不想付 C++ 的内存安全代价，还要能方便地发 Python wheel。

### 1. 所有权与 move：比 C++ move 更严格

规则（[Book ch4.1](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html)）：每个值有且只有一个 owner；owner 离开作用域时自动 `drop`（相当于析构，RAII 是默认行为）；赋值 / 传参默认是 **move**。

```rust
fn consume(v: Vec<i32>) -> usize {
    v.len() // v 在这里离开作用域，堆内存被 drop
}

fn main() {
    let a = vec![1, 2, 3];
    let b = a; // move：只拷贝 (ptr, len, cap) 三个字，所有权转给 b
    // println!("{:?}", a); // error[E0382]: borrow of moved value: `a`
    let n = consume(b); // 再 move 进函数
    let c = vec![4, 5];
    let d = c.clone(); // 深拷贝必须显式写 clone()
    let x = 1;
    let y = x; // i32 实现了 Copy：按位复制，x 仍可用
    println!("{n} {c:?} {d:?} {x} {y}");
}
```

| | C++ | Rust |
| --- | --- | --- |
| 默认语义 | 拷贝（调拷贝构造） | move（按位 memcpy） |
| move 后原对象 | 「有效但未指定」，还能用，用错是运行时 bug | 编译器标记为已移走，再用直接编译错误 |
| move 实现 | 用户写移动构造，可能有逻辑 | 不能自定义，永远是 memcpy，所以没有 `noexcept` 移动构造之类的问题 |
| 深拷贝 | 隐式（`auto b = a;`） | 显式 `.clone()` |
| 廉价复制 | 平凡可拷贝类型 | 实现 `Copy` 的类型（整数、`&T` 等） |
| 析构 | 析构函数 | `Drop` trait；被 move 走的值不会再 drop，没有 double free |

### 2. 借用规则：多个 `&T` 或一个 `&mut T`

引用叫「借用」，不拿所有权（[Book ch4.2](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html)）。同一时刻对同一数据：

- 任意多个共享引用 `&T`（只读），**或者**
- 恰好一个可变引用 `&mut T`（独占可写），
- 且引用不能活得比被引用的值久。

```rust
fn sum(v: &[i32]) -> i32 {
    v.iter().sum()
}

fn push_one(v: &mut Vec<i32>) {
    v.push(1);
}

fn main() {
    let mut v = vec![1, 2];
    let r1 = &v;
    let r2 = &v; // 多个 &T 同时存在：OK
    println!("{} {}", sum(r1), sum(r2));
    push_one(&mut v); // r1/r2 之后不再使用，借用已结束（NLL），可以拿 &mut
    let first = &v[0];
    // v.push(3); // error[E0502]: cannot borrow `v` as mutable because it is also borrowed as immutable
    println!("{first}");
}
```

被注释掉的那行在 C++ 里就是经典的迭代器 / 引用失效：`push_back` 触发扩容，`first` 指向已释放的内存。Rust 在编译期拒绝。

**为什么这能防数据竞争**：数据竞争的定义是「两个线程并发访问同一内存、至少一个是写、且没有同步」。「共享不可变、可变不共享」正好排除了「有共享又有写」这一种组合；再配合 `Send` / `Sync`（见第 7 节）把这条规则推广到跨线程，safe Rust 里就写不出数据竞争。注意它防的是 data race，不防死锁和逻辑上的竞态条件（race condition）。

### 3. 生命周期：`'a` 是什么

每个引用都有生命周期，即它有效的那段代码范围。绝大多数时候编译器自己推，只有当返回的引用可能来自多个输入时才需要标注（[Book ch10.3](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html)）。`'a` 不改变任何值活多久，只是**描述引用之间的约束**，让编译器检查。

```rust
// 返回值的生命周期 = 两个输入里较短的那个
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() >= y.len() { x } else { y }
}

// 省略规则：只有一个输入引用 → 输出自动用它的生命周期
fn first_word(s: &str) -> &str {
    s.split(' ').next().unwrap_or("")
}

// 结构体持有引用：实例不能活得比 text 久
struct Cursor<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn rest(&self) -> &'a str {
        &self.text[self.pos..]
    }
}

// fn dangle() -> &String { let s = String::new(); &s }
// error[E0106]: missing lifetime specifier —— 本质是返回了指向局部变量的引用

fn main() {
    let a = String::from("hello world");
    let w;
    {
        let b = String::from("hi");
        w = longest(&a, &b);
        println!("{w}");
    }
    // println!("{w}"); // 若放开：error[E0597]: `b` does not live long enough
    let c = Cursor { text: &a, pos: 6 };
    println!("{} {}", first_word(&a), c.rest());
}
```

**省略规则**（lifetime elision）有三条：① 每个输入引用各得一个生命周期参数；② 只有一个输入生命周期时，它赋给所有输出；③ 方法里有 `&self` / `&mut self` 时，`self` 的生命周期赋给所有输出。三条用完还确定不了输出的生命周期，就报 E0106，要求手写。`'static` 表示整个程序期间都有效（如字符串字面量），`tokio::spawn` 要求 future 是 `'static` 就是这个意思：不能借用调用者栈上的数据。

### 4. `Option` / `Result` 和 `?`：用返回值代替异常

Rust 没有异常和空指针。「可能没有」用 `Option<T>`（`Some(x)` / `None`），「可能失败」用 `Result<T, E>`（`Ok(x)` / `Err(e)`）。`?` 运算符：`Ok` 就取出值继续，`Err` 就先经过 `From` 转换再提前 `return`（[Book ch9.2](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html)）。

```rust
use std::collections::HashMap;
use std::num::ParseIntError;

#[derive(Debug)]
enum ConfigError {
    Io(std::io::Error),
    Parse(ParseIntError),
}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self { ConfigError::Io(e) }
}
impl From<ParseIntError> for ConfigError {
    fn from(e: ParseIntError) -> Self { ConfigError::Parse(e) }
}

fn token_id(vocab: &HashMap<String, u32>, tok: &str) -> Option<u32> {
    vocab.get(tok).copied() // 可能没有：Option 而不是 nullptr / -1
}

fn read_max_batch(path: &str) -> Result<u32, ConfigError> {
    let s = std::fs::read_to_string(path)?; // Err 时 From 转换后提前 return
    let n = s.trim().parse::<u32>()?;
    Ok(n)
}

fn main() {
    let vocab = HashMap::from([("hello".to_string(), 7u32)]);
    let unk = token_id(&vocab, "xyz").unwrap_or(0); // 缺省值
    match read_max_batch("/nonexistent") {
        Ok(n) => println!("max_batch={n}"),
        Err(e) => println!("fallback, err={e:?} unk={unk}"),
    }
}
```

| | C++ 异常 | Rust `Result` |
| --- | --- | --- |
| 函数签名 | 看不出会不会抛 | 返回类型里写明 |
| 忘记处理 | 往上冒，可能到 `terminate` | `Result` 带 `#[must_use]`，不处理有警告；`match` 必须覆盖所有分支 |
| 传播 | 自动栈展开 | 显式 `?`，一个字符 |
| 开销 | 零开销路径 + 抛出时很贵 | 就是普通返回值 |

不可恢复的 bug（越界、`unwrap()` 到 `None`）走 `panic!`，默认展开当前线程。工程里常用 `thiserror` 定义库的错误类型、`anyhow` 在应用层收拢各种错误。

### 5. trait 与泛型：单态化 vs `dyn Trait`

trait 类似 C++20 concept + 抽象基类的合体：定义一组方法，类型通过 `impl Trait for Type` 实现（[Book ch10.2](https://doc.rust-lang.org/book/ch10-02-traits.html)）。用法有两种：

```rust
trait Sampler {
    fn sample(&self, logits: &[f32]) -> usize;
}

struct Greedy;
impl Sampler for Greedy {
    fn sample(&self, logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

struct ForceEos { eos: usize }
impl Sampler for ForceEos {
    fn sample(&self, _logits: &[f32]) -> usize { self.eos }
}

// 静态分发：编译器为每个具体 S 生成一份代码（单态化），可内联
fn step_static<S: Sampler>(s: &S, logits: &[f32]) -> usize {
    s.sample(logits)
}

// 动态分发：&dyn Sampler 是胖指针 (data_ptr, vtable_ptr)
fn step_dyn(s: &dyn Sampler, logits: &[f32]) -> usize {
    s.sample(logits)
}

fn main() {
    let logits = [0.1, 2.0, 0.3];
    let a = step_static(&Greedy, &logits);
    // 运行时才确定类型的异构集合，只能用 trait object
    let samplers: Vec<Box<dyn Sampler>> = vec![Box::new(Greedy), Box::new(ForceEos { eos: 2 })];
    let b: Vec<usize> = samplers.iter().map(|s| step_dyn(s.as_ref(), &logits)).collect();
    assert_eq!(std::mem::size_of::<&dyn Sampler>(), 2 * std::mem::size_of::<usize>());
    println!("{a} {b:?}");
}
```

| | C++ | Rust |
| --- | --- | --- |
| 静态多态 | 模板，实例化时才检查（C++20 前是鸭子类型） | 泛型 + trait bound，**定义处**就按 bound 检查，报错清晰；同样单态化（[Book ch10.1](https://doc.rust-lang.org/book/ch10-01-syntax.html)），代价是编译时间和二进制膨胀 |
| 动态多态 | 虚函数；vptr 存在**对象里** | `dyn Trait`；vtable 指针存在**引用 / Box 里**（胖指针，两个字长），对象本身不带 vptr（[Book ch18.2](https://doc.rust-lang.org/book/ch18-02-trait-objects.html)） |
| 继承 | 有，可多继承 | 没有数据继承；只有 trait 组合和 trait 之间的 supertrait 约束 |
| 给已有类型加接口 | 不行（只能包一层） | 可以给外部类型 `impl` 自己的 trait（受孤儿规则限制） |

胖指针的好处：同一个类型可以以不同 trait 的身份出现，不需要在对象布局里为每个基类放 vptr。

### 6. 智能指针与内部可变性：对照 C++

| Rust | C++ 近似 | 要点 |
| --- | --- | --- |
| `&T` / `&mut T` | `const T&` / `T&` | 但受借用规则和生命周期检查 |
| `Box<T>` | `std::unique_ptr<T>` | 堆上独占，零开销；不会为空（要空用 `Option<Box<T>>`，大小不变） |
| `Rc<T>` | `std::shared_ptr<T>`（非原子计数版） | 单线程引用计数，计数用普通加减，比 `Arc` 便宜；**不能跨线程**（不是 `Send`） |
| `Arc<T>` | `std::shared_ptr<T>` | 原子引用计数，跨线程共享；内部默认只读 |
| `Weak<T>` | `std::weak_ptr<T>` | 打破 `Rc` / `Arc` 循环引用 |
| `Cell<T>` / `RefCell<T>` | `mutable` 成员 | 内部可变性：通过 `&T` 修改；`RefCell` 把借用检查推迟到运行时，违规就 panic（[Book ch15.5](https://doc.rust-lang.org/book/ch15-05-interior-mutability.html)） |
| `Mutex<T>` | `std::mutex` + 被保护的数据 | 锁**包住数据**，不 `lock()` 拿不到数据；`MutexGuard` 析构自动解锁（相当于 `lock_guard`） |
| `RwLock<T>` | `std::shared_mutex` | 读多写少 |
| `AtomicU64` 等 | `std::atomic<uint64_t>` | 内存序也是 `Relaxed / Acquire / Release / SeqCst` 那一套 |

组合惯用法：单线程共享可变用 `Rc<RefCell<T>>`，多线程共享可变用 `Arc<Mutex<T>>`（[Book ch16.3](https://doc.rust-lang.org/book/ch16-03-shared-state.html)）。

```rust
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    // Rc<RefCell<T>>：单线程共享 + 运行时借用检查
    let cache = Rc::new(RefCell::new(Vec::<u32>::new()));
    let c2 = Rc::clone(&cache); // 只加计数（非原子）
    c2.borrow_mut().push(1);
    {
        let r = cache.borrow();
        // let w = cache.borrow_mut(); // 编译通过，但运行时 panic：RefCell already borrowed
        println!("len={} rc={}", r.len(), Rc::strong_count(&cache));
    }

    // Arc<Mutex<T>>：跨线程共享可变状态
    let stats = Arc::new(Mutex::new(0u64));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let s = Arc::clone(&stats);
            thread::spawn(move || {
                *s.lock().unwrap() += 1; // MutexGuard 离开作用域自动解锁
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    println!("{}", *stats.lock().unwrap());

    // let rc = Rc::new(5);
    // thread::spawn(move || println!("{rc}"));
    // error[E0277]: `Rc<i32>` cannot be sent between threads safely
}
```

`lock()` 返回 `Result`，是因为持锁线程 panic 后锁会被标记为 poisoned，提醒你数据可能处于不一致状态。

### 7. `Send` / `Sync`：线程安全写进类型

两个标记 trait（marker trait，没有方法，[Book ch16.4](https://doc.rust-lang.org/book/ch16-04-extensible-concurrency-sync-and-send.html)、[Nomicon](https://doc.rust-lang.org/nomicon/send-and-sync.html)）：

- `T: Send`：`T` 的**所有权**可以转移到另一个线程。
- `T: Sync`：`&T` 可以在多个线程间共享，等价于 `&T: Send`。

它们是 auto trait：一个类型的所有字段都是 `Send`，它就自动是 `Send`。几个关键反例：

| 类型 | Send | Sync | 原因 |
| --- | --- | --- | --- |
| `Rc<T>` | 否 | 否 | 计数非原子，两个线程同时 clone 会算错 |
| `Arc<T>`（`T: Send + Sync`） | 是 | 是 | 原子计数 |
| `RefCell<T>` / `Cell<T>` | 是（`T: Send`） | 否 | 运行时借用标记非线程安全 |
| `Mutex<T>`（`T: Send`） | 是 | 是 | 锁提供同步 |
| `*const T` / `*mut T` | 否 | 否 | 编译器不知道指向什么 |
| `MutexGuard<T>` | 否 | 是（`T: Sync`） | 有些平台要求在加锁的线程上解锁 |

`thread::spawn` 的签名要求闭包 `F: Send + 'static`，所以把 `Rc` move 进去直接报 E0277。**线程安全不是文档约定，而是函数签名上的约束**。给自己的类型手写 `unsafe impl Send` 意味着你向编译器担保，写错就是 UB。

### 8. async/await 与 tokio

`async fn` 返回一个实现了 `Future` 的状态机。它是**惰性的**：不被 poll 就什么都不做；标准库只定义 `Future`，不带运行时，需要 tokio 这样的 executor 来驱动（[tokio：Async in depth](https://tokio.rs/tokio/tutorial/async)）。

```rust
use std::time::Duration;

async fn fetch(id: u32) -> u32 {
    tokio::time::sleep(Duration::from_millis(10)).await; // 让出线程，不阻塞
    id * 2
}

#[tokio::main] // 宏展开 = 建一个多线程 Runtime，然后 block_on(main 的 future)
async fn main() {
    let fut = fetch(1); // 只构造了状态机，什么都没执行
    let a = fut.await; // 被 poll 才开始跑

    let (b, c) = tokio::join!(fetch(2), fetch(3)); // 同一个 task 内并发等待

    let h = tokio::spawn(fetch(4)); // 交给运行时，可能在别的 worker 线程上跑
    let d = h.await.unwrap();

    // CPU 密集 / 同步阻塞调用：挪到专门的阻塞线程池
    let e = tokio::task::spawn_blocking(|| (0..1_000_000u64).sum::<u64>())
        .await
        .unwrap();
    println!("{a} {b} {c} {d} {e}");
}
```

| | Python asyncio | Rust + tokio |
| --- | --- | --- |
| 协程对象 | `async def` 返回 coroutine，也要 `await` 或 `create_task` 才跑 | `async fn` 返回 Future，要 `.await` 或 `spawn` 才跑 |
| 事件循环 | 标准库内置，单线程 | 第三方 runtime；默认多线程 work-stealing |
| 调度单位 | `Task` | task（`tokio::spawn`），要求 `Send + 'static`，因为可能被挪到别的线程 |
| 阻塞调用的坑 | 同步慢函数卡住整个 loop → `run_in_executor` | 同样会卡住 worker 线程 → `spawn_blocking` |
| 并发等待 | `asyncio.gather` | `tokio::join!` / `JoinSet` |
| 有界队列 | `asyncio.Queue(maxsize)` | `tokio::sync::mpsc::channel(cap)`（[tokio：Channels](https://tokio.rs/tokio/tutorial/channels)） |

两个常见坑：① 不要把 `std::sync::MutexGuard` 跨 `.await` 持有，会让 future 变成非 `Send` 且可能死锁，要么缩小临界区，要么用 `tokio::sync::Mutex`（[tokio：Shared state](https://tokio.rs/tokio/tutorial/shared-state)）；② tokio 是协作式调度，长时间不 `.await` 的计算会饿死同一 worker 上的其他 task。

### 9. `unsafe` 与 FFI

`unsafe` 不是关掉借用检查，而是额外解锁五种操作：解引用裸指针、调用 `unsafe fn`（包括 FFI 函数）、访问可变 static、实现 `unsafe trait`（如 `Send`）、读 union 字段（[Book ch20.1](https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html)）。惯用法是把 unsafe 包在小范围里，对外暴露 safe API，并写 `// SAFETY:` 注释说明不变量。

```rust
// 调用 C：Rust 1.82+ 写法，extern 块本身标 unsafe
unsafe extern "C" {
    fn abs(x: i32) -> i32;
}

// 导出给 C / Python ctypes 调用
#[unsafe(no_mangle)]
pub extern "C" fn rs_sum(ptr: *const f32, len: usize) -> f32 {
    if ptr.is_null() {
        return 0.0;
    }
    // SAFETY: 调用方保证 ptr 指向 len 个有效、对齐的 f32，且调用期间不被修改
    let xs = unsafe { std::slice::from_raw_parts(ptr, len) };
    xs.iter().sum()
}

fn main() {
    let v = [1.0f32, 2.0, 3.0];
    let s = rs_sum(v.as_ptr(), v.len());
    let a = unsafe { abs(-3) };
    println!("{s} {a}");
}
```

FFI 边界上要注意的点（[Nomicon：FFI](https://doc.rust-lang.org/nomicon/ffi.html)）：结构体加 `#[repr(C)]` 才有 C 兼容布局；panic 不应跨 FFI 边界展开（`extern "C"` 函数里 panic 会直接 abort，需要时用 `catch_unwind` 转成错误码）；谁分配谁释放（Rust 分配的内存交还给 Rust 的函数释放）。调 CUDA / C++ 库时一般用 `bindgen` 生成声明，或 `cxx` crate 做 C++ 互操作。

### 10. PyO3：给 Python 写扩展并释放 GIL

PyO3 用宏把 Rust 函数 / 结构体暴露为 Python 模块，`maturin` 负责编译打包成 wheel（[PyO3 入门](https://pyo3.rs/main/getting-started.html)、[maturin](https://www.maturin.rs/)）。tokenizers、safetensors 的 Python 包都是这么做的。

```rust
use pyo3::prelude::*;

/// 纯 Rust 逻辑：不碰任何 Python 对象
fn count_tokens(texts: &[String]) -> Vec<usize> {
    texts.iter().map(|t| t.split_whitespace().count()).collect()
}

#[pyfunction]
fn batch_count(py: Python<'_>, texts: Vec<String>) -> Vec<usize> {
    // texts 已从 Python list 拷贝成 Vec<String>（此时持有 GIL）
    // detach：释放 GIL 跑纯 Rust 计算，其他 Python 线程（如 asyncio 事件循环）可以继续
    py.detach(|| count_tokens(&texts))
}

#[pymodule]
fn fasttok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_count, m)?)?;
    Ok(())
}
```

- **API 名字**：PyO3 0.26 起 `py.allow_threads(...)` 改名为 `py.detach(...)`（`with_gil` 改名为 `attach`），因为 free-threaded CPython 没有 GIL 了，统一叫「从解释器分离」。老代码和很多文章里还是 `allow_threads`（[PyO3：Parallelism](https://pyo3.rs/main/parallelism)）。
- **为什么要释放**：持有 GIL 跑 100 ms 的分词，同进程的其他 Python 线程全被卡住；释放后 API server 的事件循环能继续接请求，Rust 侧还能用 Rayon 开多线程。
- **释放期间不能碰 Python 对象**：闭包要求 `Ungil`（大致等于 `Send`），编译器会拦住把 `Bound<PyAny>` 带进去，所以先把输入转成 Rust 类型（上面的 `Vec<String>`）。
- **实际例子**：tokenizers 的 Python 绑定里，`encode`、`encode_batch`、`decode` 都包在 `py.detach(...)` 里（[源码](https://github.com/huggingface/tokenizers/blob/main/bindings/python/src/tokenizer.rs)）。

### 11. cargo 基础

cargo 是构建工具 + 包管理器 + 测试运行器（[Cargo Book](https://doc.rust-lang.org/cargo/getting-started/)），大致相当于 CMake + pip + pytest 合一。

```toml
# Cargo.toml
[package]
name = "fasttok"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]   # 编成动态库供 Python 加载

[dependencies]
pyo3 = { version = "0.26", features = ["extension-module"] }
tokio = { version = "1", features = ["full"] }

[profile.release]
lto = true
```

| 命令 | 作用 |
| --- | --- |
| `cargo new / init` | 建项目（`--lib` 建库） |
| `cargo build --release` | 优化构建；不加 `--release` 是 debug 构建，慢很多，性能测试别用 |
| `cargo run / test / bench` | 运行、跑单元测试（`#[test]`）、基准测试 |
| `cargo check` | 只做类型 / 借用检查不生成代码，迭代最快 |
| `cargo clippy / fmt` | lint 和格式化 |
| `cargo add <crate>` | 加依赖 |

概念：crate 是编译单元，package 包含一个或多个 crate；`Cargo.lock` 锁定依赖版本；workspace 管理多 crate 仓库（如 TGI 的 router、launcher、backends 是同一个 workspace）；features 是条件编译开关（如 candle 的 `cuda`）。

## 面试追问

::: details Q：Rust 怎么在编译期防止数据竞争？
三层机制叠加。① 借用规则：同一时刻要么多个 `&T` 要么一个 `&mut T`，所以不可能「有人在读、同时有人在写」而没有同步。② `Send` / `Sync`：跨线程的 API（`thread::spawn`、`tokio::spawn`）在签名上要求 `Send`，共享引用要求 `Sync`；`Rc`、`RefCell`、裸指针这些非线程安全的类型不满足，传过去直接编译错误。③ 共享可变状态只能通过 `Mutex` / `RwLock` / 原子类型这类实现了 `Sync` 的内部可变性容器，而 `Mutex<T>` 把数据包在锁里面，不加锁根本拿不到 `&mut T`。边界：防的是 data race，不防死锁、不防逻辑竞态（比如 check-then-act），`unsafe` 代码里也需要自己保证。
:::

::: details Q：Arc&lt;Mutex&lt;T&gt;&gt; 和 C++ 的 shared_ptr + mutex 有什么区别？
- 引用计数部分 `Arc` ≈ `shared_ptr`，都是原子计数，开销一致。区别是 Rust 还有非原子的 `Rc`，并且编译器保证 `Rc` 不会被误用到多线程。
- 锁和数据的关系：C++ 里 `mutex` 和数据是两个独立成员，「访问 `data` 前要先锁 `mu`」只是约定，忘了加锁照样能编译；Rust 的 `Mutex<T>` 拥有数据，唯一的访问途径是 `lock()` 返回的 `MutexGuard`，guard 析构时解锁，借用检查保证引用不会逃出 guard 的作用域。
- 共享可变：`Arc<T>` 只给 `&T`，想改必须配内部可变性（`Mutex`、`RwLock`、原子类型）；`shared_ptr<T>` 可以直接拿到非 const 指针随便改。
- 毒化：持锁线程 panic 后锁被标记 poisoned，后续 `lock()` 返回 `Err`；C++ 没有这个概念。
- 单个计数器这种场景两边都应该用原子变量而不是锁。
:::

::: details Q：为什么 tokenizer 用 Rust 写，而不是纯 Python 或 C++？
- 相对纯 Python：分词（正则预切分、BPE merge、offset 对齐）是 CPU 密集的字符串处理，纯 Python 实现慢得多，且受 GIL 限制不能多线程。Rust 实现可以在释放 GIL 后用 Rayon 并行处理一个 batch。
- 在推理服务里尤其重要：API server 是 asyncio 单线程事件循环，如果 tokenize 持有 GIL，长 prompt 会卡住所有请求的收发；tokenizers 在 `encode` / `encode_batch` 外层调用 `py.detach`，计算期间事件循环和其他线程照常运行。
- 相对 C++：性能同级，但内存安全由编译器保证（处理任意用户输入的字符串解析代码最容易出越界）；PyO3 + maturin 打 wheel 比 pybind11 + CMake 简单；同一套 Rust 核心还能出 Node、Ruby 等绑定，也能被 Rust 写的 router / gateway（TGI、SGLang Model Gateway）直接复用，不经过 Python。
:::

::: details Q：Rust 的 move 和 C++ 的 move 有什么本质区别？
C++ 的 move 是一次函数调用（移动构造 / 移动赋值），源对象之后仍然存在、仍会被析构，所以必须留在「有效但未指定」的状态，用了也不报错。Rust 的 move 是语言层面的所有权转移：永远是按位拷贝，源变量被编译器标记为不可用，不会再被 drop，使用它是编译错误。所以 Rust 不需要移动构造函数，也没有「被移走的对象」这种状态；需要显式拷贝时用 `Clone`，平凡可复制的类型实现 `Copy`。
:::

::: details Q：safe Rust 会内存泄漏吗？
会。内存泄漏不算内存不安全。`Rc` / `Arc` 循环引用会泄漏（要用 `Weak` 打破环，和 `shared_ptr` 一样）；`std::mem::forget`、`Box::leak` 也是 safe 函数。Rust 保证的是没有悬垂指针、double free、数据竞争，不保证一定释放。
:::

::: details Q：tokio::spawn 为什么要求 Send + 'static？
多线程运行时会把 task 在不同 worker 线程之间迁移（work-stealing），所以 future 本身和它跨 `.await` 保存的所有状态都必须是 `Send`；task 的生命周期由运行时管理，可能比调用者活得更久，所以不能借用调用者栈上的数据，必须是 `'static`（把数据 move 进去或用 `Arc` 共享）。如果确实要用非 `Send` 的东西，可以用 `LocalSet` / `spawn_local` 把 task 固定在一个线程上。
:::

::: details Q：泛型（单态化）和 dyn Trait 怎么选？
默认用泛型：零开销、可内联，适合热路径（比如对每个 token 调用的 sampler）。以下情况用 `dyn Trait`：需要异构集合（`Vec<Box<dyn Sampler>>`）、要在运行时根据配置选实现、要控制编译时间和二进制大小、插件式扩展。代价是一次间接调用和无法内联，和 C++ 虚函数一样。另外 trait 需要满足 dyn 兼容（比如方法不能是泛型、不能返回 `Self`）才能做成 trait object。
:::

## 手撕

### 线程安全计数器（Mutex 版和原子版）

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

// 版本 1：Mutex —— 能保护任意复合状态
#[derive(Clone, Default)]
struct Counter {
    inner: Arc<Mutex<u64>>,
}

impl Counter {
    fn incr(&self) {
        *self.inner.lock().unwrap() += 1;
    }
    fn get(&self) -> u64 {
        *self.inner.lock().unwrap()
    }
}

// 版本 2：原子变量 —— 单个整数用它更快
#[derive(Default)]
struct AtomicCounter {
    n: AtomicU64,
}

impl AtomicCounter {
    fn incr(&self) {
        self.n.fetch_add(1, Ordering::Relaxed); // 纯计数不需要和其他内存操作建立顺序
    }
    fn get(&self) -> u64 {
        self.n.load(Ordering::Relaxed)
    }
}

fn main() {
    let c = Counter::default();
    let hs: Vec<_> = (0..8)
        .map(|_| {
            let c = c.clone();
            thread::spawn(move || (0..1000).for_each(|_| c.incr()))
        })
        .collect();
    hs.into_iter().for_each(|h| h.join().unwrap());
    assert_eq!(c.get(), 8000);

    let ac = AtomicCounter::default();
    thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| (0..1000).for_each(|_| ac.incr())); // scoped thread 可直接借用栈上的 ac
        }
    });
    assert_eq!(ac.get(), 8000);
    println!("ok");
}
```

讲解点：`incr(&self)` 只要共享引用就能改，因为可变性在 `Mutex` / 原子类型内部；`thread::scope` 保证所有子线程在作用域结束前 join，所以可以借用而不必 `Arc`。

### 有界队列：生产者-消费者（Mutex + Condvar 手写）

```rust
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

pub struct BoundedQueue<T> {
    buf: Mutex<VecDeque<T>>,
    not_full: Condvar,
    not_empty: Condvar,
    cap: usize,
}

impl<T> BoundedQueue<T> {
    pub fn new(cap: usize) -> Self {
        Self {
            buf: Mutex::new(VecDeque::with_capacity(cap)),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
            cap,
        }
    }

    pub fn push(&self, item: T) {
        let mut q = self.buf.lock().unwrap();
        // while 而不是 if：防虚假唤醒
        while q.len() == self.cap {
            q = self.not_full.wait(q).unwrap();
        }
        q.push_back(item);
        self.not_empty.notify_one();
    }

    pub fn pop(&self) -> T {
        let mut q = self.buf.lock().unwrap();
        while q.is_empty() {
            q = self.not_empty.wait(q).unwrap();
        }
        let item = q.pop_front().unwrap();
        self.not_full.notify_one();
        item
    }
}

fn main() {
    let q = Arc::new(BoundedQueue::new(4));
    let producers: Vec<_> = (0..2)
        .map(|p| {
            let q = Arc::clone(&q);
            thread::spawn(move || {
                for i in 0..100 {
                    q.push(Some(p * 1000 + i));
                }
            })
        })
        .collect();
    let consumer = {
        let q = Arc::clone(&q);
        thread::spawn(move || {
            let (mut n, mut done) = (0, 0);
            while done < 2 {
                match q.pop() {
                    Some(_) => n += 1,
                    None => done += 1, // 每个生产者结束时发一个 None 作为哨兵
                }
            }
            n
        })
    };
    for h in producers {
        h.join().unwrap();
    }
    q.push(None);
    q.push(None);
    assert_eq!(consumer.join().unwrap(), 200);
    println!("ok");
}
```

讲解点：`Condvar::wait` 接收 guard、原子地释放锁并睡眠，醒来时重新拿到锁并返回新 guard，和 C++ `condition_variable::wait(unique_lock&)` 一样；容量满时 `push` 阻塞就是背压（推理服务里请求队列满了让上游等待或拒绝）。

实际工程里直接用标准库的有界通道 [`mpsc::sync_channel`](https://doc.rust-lang.org/std/sync/mpsc/fn.sync_channel.html)，结束信号靠 drop 所有 `Sender`，不需要哨兵：

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::sync_channel::<u32>(4); // 容量 4，满了 send 阻塞 = 背压
    let producers: Vec<_> = (0..2)
        .map(|p| {
            let tx = tx.clone();
            thread::spawn(move || {
                for i in 0..100 {
                    tx.send(p * 1000 + i).unwrap();
                }
            })
        })
        .collect();
    drop(tx); // 所有 Sender drop 后，rx 迭代自动结束
    let n = rx.iter().count();
    producers.into_iter().for_each(|h| h.join().unwrap());
    assert_eq!(n, 200);
    println!("ok");
}
```

async 版本把 `sync_channel` 换成 `tokio::sync::mpsc::channel(cap)`，`send(x).await` 在满时挂起当前 task 而不是阻塞线程。

## 参考

- [The Rust Programming Language（Book）](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html)：所有权 ch4、错误处理 ch9、泛型 / trait / 生命周期 ch10、智能指针 ch15、并发 ch16、trait object ch18、unsafe 与 FFI ch20
- [The Rustonomicon：Send and Sync](https://doc.rust-lang.org/nomicon/send-and-sync.html)、[FFI](https://doc.rust-lang.org/nomicon/ffi.html)
- [Tokio 教程：Async in depth](https://tokio.rs/tokio/tutorial/async)、[Shared state](https://tokio.rs/tokio/tutorial/shared-state)、[Channels](https://tokio.rs/tokio/tutorial/channels)
- [PyO3 用户指南：Parallelism](https://pyo3.rs/main/parallelism)
- [The Cargo Book](https://doc.rust-lang.org/cargo/getting-started/)
- 项目：[huggingface/tokenizers](https://github.com/huggingface/tokenizers)、[huggingface/safetensors](https://github.com/huggingface/safetensors)、[TGI 架构](https://huggingface.co/docs/text-generation-inference/architecture)、[SGLang Model Gateway](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)、[huggingface/candle](https://github.com/huggingface/candle)

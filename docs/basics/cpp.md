---
title: C++：RAII、智能指针、移动语义、虚函数表
status: todo
tags: [cpp, raii]
difficulty: 2
order: 2
related: []
stack: []
---

# C++：RAII、智能指针、移动语义、虚函数表

> kernel / framework 岗常问

## 一句话结论

C++ 面试在 infra 岗考的是「资源归谁管、什么时候释放、调用开销在哪」：RAII 把资源生命周期绑到栈对象上，智能指针把所有权写进类型，移动语义避免深拷贝，虚函数表是运行时多态的代价。

## 推导

- **RAII**：构造获取、析构释放，异常路径也能正确释放；`std::lock_guard`、CUDA stream 的 wrapper 都是这个模式。
- **智能指针**：`unique_ptr` 独占、零开销；`shared_ptr` 引用计数（原子操作，有开销），`weak_ptr` 打破环；裸指针只做「借用」。
- **移动语义**：右值引用 + `std::move` 让容器扩容、函数返回大对象时只搬指针不拷数据；被移走的对象处于「有效但未指定」状态。
- **虚函数表**：每个多态类一张 vtable，对象里存一个 vptr；虚调用多一次间接跳转且阻碍内联，这是 ATen dispatcher 用函数指针表而不是虚函数的原因之一。

## 面试追问

::: details Q：shared_ptr 的引用计数为什么是原子的，开销有多大？
因为多个线程可能同时拷贝 / 析构同一个 `shared_ptr`，计数必须用原子增减。单次原子操作几十纳秒，热路径里频繁拷贝 `shared_ptr` 会明显拖慢；常见做法是函数参数按 `const shared_ptr&` 或裸指针传递，只在真正需要延长生命周期时拷贝。
:::

## 手撕

常见题：手写一个简化版 `unique_ptr`（禁止拷贝、允许移动、析构释放）；或写一个 RAII 的 scoped timer。

## 参考

- [C++ Core Guidelines：资源管理一节](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-resource)
- [cppreference：std::unique_ptr](https://en.cppreference.com/w/cpp/memory/unique_ptr)

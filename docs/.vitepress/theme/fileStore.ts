import { ref, shallowRef, type Ref } from 'vue'

/**
 * 一份存在仓库 json 里的数据（闪卡的复习记录、批注）：
 * dev 时通过 dev server 的接口读写文件，随 git 同步；线上文件打包进站点只读，本机的改动存 localStorage，
 * 加载时用 merge 合并（按条目取较新的）。和 ReaderNotes 是同一套做法。
 */
export interface FileStoreOptions<T> {
  /** 日志前缀 */
  label: string
  endpoint: string
  localKey: string
  bundled: T
  validate: (v: unknown) => v is T
  /** over 里较新的条目盖过 base */
  merge: (base: T, over: T) => T
}

export interface FileStore<T> {
  data: Ref<T>
  /** 是否以仓库文件为数据源（dev 且 dev server 可达） */
  fileMode: Ref<boolean>
  load: () => Promise<void>
  /** 把当前的 data 存下来；写文件串行，保证落盘顺序和操作顺序一致 */
  persist: () => void
}

export function useFileStore<T extends object>(o: FileStoreOptions<T>): FileStore<T> {
  const data = shallowRef<T>(o.bundled) as Ref<T>
  const fileMode = ref(import.meta.env.DEV)

  function loadLocal(): T {
    try {
      const v = JSON.parse(localStorage.getItem(o.localKey) || '{}')
      return o.validate(v) ? v : ({} as T)
    } catch {
      return {} as T
    }
  }

  function saveLocal(v: T) {
    try {
      localStorage.setItem(o.localKey, JSON.stringify(v))
    } catch {}
  }

  async function putFile(v: T) {
    const res = await fetch(o.endpoint, { method: 'PUT', body: JSON.stringify(v) })
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  }

  async function read(): Promise<T> {
    if (!fileMode.value) return o.merge(o.bundled, loadLocal())
    try {
      const res = await fetch(o.endpoint)
      if (!res.ok) throw new Error(`${res.status}`)
      const file = await res.json()
      if (!o.validate(file)) throw new Error('文件格式不对')
      // 浏览器里的旧数据（或之前写文件失败暂存的）并入文件，成功后清掉本地副本
      const local = loadLocal()
      if (!Object.keys(local).length) return file
      const merged = o.merge(file, local)
      await putFile(merged)
      try { localStorage.removeItem(o.localKey) } catch {}
      return merged
    } catch (e) {
      console.error(`[${o.label}] 读不到仓库文件，退回 localStorage`, e)
      fileMode.value = false
      return o.merge(o.bundled, loadLocal())
    }
  }

  let saving: Promise<void> = Promise.resolve()

  return {
    data,
    fileMode,
    load: async () => { data.value = await read() },
    persist() {
      const snapshot = data.value
      if (!fileMode.value) return saveLocal(snapshot)
      saving = saving
        .then(() => putFile(snapshot))
        .catch((e) => {
          console.error(`[${o.label}] 写仓库文件失败，暂存到 localStorage`, e)
          saveLocal(snapshot)
        })
    },
  }
}

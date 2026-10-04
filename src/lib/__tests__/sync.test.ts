import { describe, expect, it } from 'vitest'
import {
  addNote,
  editNote,
  isReaderNotes,
  liveNotes,
  mergeReaderNotes,
  removeNotes,
  serializeReaderNotes,
  type ReaderNote,
  type ReaderNotes,
} from '@lib/readerNotes'
import { InvalidRemoteError, StaleError, syncDocs, type Remote, type RemoteFile, type SyncTarget } from '@lib/sync'
import { SYNC_DOCS, cardNotesSpec, readerNotesSpec } from '@lib/syncDocs'
import { AuthError, githubRemote, readQuery, toBase64, type RepoConfig } from '@lib/github'
import { setNote, type Notes } from '@lib/flashcards'
import readerNotesFile from '@data/reader-notes.json'

const P = '/ai-infra-inferview/handson/x'
const note = (id: string, created: number, extra: Partial<ReaderNote> = {}): ReaderNote => ({
  id,
  exact: `quote ${id}`,
  prefix: '',
  suffix: '',
  note: '',
  created,
  ...extra,
})

describe('reader notes', () => {
  it('仓库里的批注文件格式正确，且已经是规范输出', () => {
    expect(isReaderNotes(readerNotesFile)).toBe(true)
    expect(JSON.parse(serializeReaderNotes(readerNotesFile as ReaderNotes))).toEqual(readerNotesFile)
  })

  it('按 id 合并，较新的修改胜出，老数据按 created 算', () => {
    const a: ReaderNotes = { [P]: [note('x', 1), note('y', 1)] }
    const b: ReaderNotes = { [P]: [note('x', 1, { note: 'new', at: 5 }), note('z', 2)] }
    const m = mergeReaderNotes(a, b)
    expect(m[P].map((n) => n.id).sort()).toEqual(['x', 'y', 'z'])
    expect(m[P].find((n) => n.id === 'x')!.note).toBe('new')
    // 旧副本不会盖掉新的
    expect(mergeReaderNotes(b, a)[P].find((n) => n.id === 'x')!.note).toBe('new')
  })

  it('删除留下删除记录，别的设备上的旧副本合并后不会复活', () => {
    const phone: ReaderNotes = { [P]: [note('x', 1, { note: 'hi' })] }
    const desktop = removeNotes(phone, P, ['x'], 10)
    expect(liveNotes(desktop[P])).toEqual([])
    expect(desktop[P][0]).toMatchObject({ deleted: true, note: '', at: 10 })
    expect(liveNotes(mergeReaderNotes(desktop, phone)[P])).toEqual([])
    expect(liveNotes(mergeReaderNotes(phone, desktop)[P])).toEqual([])
    // 删除记录留在文件里
    expect(JSON.parse(serializeReaderNotes(desktop))[P]).toHaveLength(1)
  })

  it('新增和编辑', () => {
    let s = addNote({}, P, note('x', 1))
    s = editNote(s, P, 'x', 'hello', 7)
    expect(s[P][0]).toMatchObject({ note: 'hello', at: 7 })
    expect(isReaderNotes(s)).toBe(true)
  })
})

/** 内存里的远端，行为模仿 GitHub：带版本号，版本对不上就拒绝 */
function fakeRemote(initial: Record<string, string | null> = {}) {
  const files: Record<string, string | null> = { ...initial }
  let version = 0
  const writes: RemoteFile[][] = []
  const remote: Remote & { files: typeof files; writes: typeof writes; bump: (path: string, text: string) => void } = {
    files,
    writes,
    bump(path, text) {
      files[path] = text
      version++
    },
    async read(paths) {
      return { texts: Object.fromEntries(paths.map((p) => [p, files[p] ?? null])), version: String(version) }
    },
    async write(changes, v) {
      if (v !== String(version)) throw new StaleError()
      writes.push(changes)
      for (const c of changes) files[c.path] = c.text
      version++
    },
  }
  return remote
}

function target<T>(spec: SyncTarget<T>['spec'], initial: T) {
  const box = { value: initial }
  const t: SyncTarget<T> = { spec, get: () => box.value, set: (v) => (box.value = v) }
  return { t, box }
}

describe('syncDocs', () => {
  it('远端没有文件时写上去；再同步一次没有变化就不写', async () => {
    const remote = fakeRemote()
    const { t } = target(cardNotesSpec, setNote({}, 'c1', 'hello', 1))
    expect((await syncDocs(remote, [t])).written).toEqual([cardNotesSpec.path])
    expect(JSON.parse(remote.files[cardNotesSpec.path]!)).toEqual({ c1: { text: 'hello', at: 1 } })
    expect((await syncDocs(remote, [t])).written).toEqual([])
    expect(remote.writes).toHaveLength(1)
  })

  it('远端较新的改动并进本地，本地较新的写回远端', async () => {
    const remote = fakeRemote({ [cardNotesSpec.path]: cardNotesSpec.serialize(setNote(setNote({}, 'a', 'remote', 5), 'b', 'old', 1)) })
    const { t, box } = target<Notes>(cardNotesSpec, setNote({}, 'b', 'local', 3))
    await syncDocs(remote, [t])
    expect(box.value).toEqual({ a: { text: 'remote', at: 5 }, b: { text: 'local', at: 3 } })
    expect(JSON.parse(remote.files[cardNotesSpec.path]!)).toEqual(box.value)
  })

  it('写之前别的设备先提交了：重读、重合并、再写，两边的改动都在', async () => {
    const remote = fakeRemote()
    const { t } = target<Notes>(cardNotesSpec, setNote({}, 'mine', 'phone', 2))
    const read = remote.read.bind(remote)
    let raced = false
    remote.read = async (paths) => {
      const snap = await read(paths)
      if (!raced) {
        raced = true
        remote.bump(cardNotesSpec.path, cardNotesSpec.serialize(setNote({}, 'theirs', 'desktop', 1)))
      }
      return snap
    }
    const r = await syncDocs(remote, [t])
    expect(r.retries).toBe(1)
    expect(Object.keys(JSON.parse(remote.files[cardNotesSpec.path]!)).sort()).toEqual(['mine', 'theirs'])
  })

  it('多份数据有变化时一次写完（GitHub 上是一个 commit）', async () => {
    const remote = fakeRemote()
    const a = target(cardNotesSpec, setNote({}, 'c', 'x', 1)).t
    const b = target(readerNotesSpec, addNote({}, P, note('n', 1))).t
    await syncDocs(remote, [a, b])
    expect(remote.writes).toHaveLength(1)
    expect(remote.writes[0].map((f) => f.path).sort()).toEqual([cardNotesSpec.path, readerNotesSpec.path].sort())
  })

  it('远端文件格式不对时不覆盖', async () => {
    const remote = fakeRemote({ [cardNotesSpec.path]: '{ broken' })
    const { t } = target(cardNotesSpec, setNote({}, 'c', 'x', 1))
    await expect(syncDocs(remote, [t])).rejects.toBeInstanceOf(InvalidRemoteError)
    expect(remote.files[cardNotesSpec.path]).toBe('{ broken')
  })

  it('同步的几份数据路径、本地 key 都不重复', () => {
    for (const k of ['key', 'path', 'localKey'] as const) expect(new Set(SYNC_DOCS.map((d) => d[k])).size).toBe(SYNC_DOCS.length)
  })
})

describe('github remote', () => {
  const repo: RepoConfig = { owner: 'o', name: 'r', branch: 'data' }
  const paths = ['src/data/a.json', 'src/data/b.json']

  /** 记下请求，按顺序回放响应 */
  function fakeFetch(responses: { status?: number; body: unknown }[]) {
    const calls: { query: string; variables: any }[] = []
    const fn = async (_url: string, init: RequestInit) => {
      calls.push(JSON.parse(init.body as string))
      const r = responses.shift()!
      return new Response(JSON.stringify(r.body), { status: r.status ?? 200 })
    }
    return { fn, calls }
  }

  const commit = (oid: string, texts: (string | null)[]) => ({
    oid,
    ...Object.fromEntries(texts.map((t, i) => [`f${i}`, t === null ? null : { object: { text: t } }])),
  })

  it('base64 能编码中文', () => {
    const s = '批注 ✎ hello'
    expect(new TextDecoder().decode(Uint8Array.from(atob(toBase64(s)), (c) => c.charCodeAt(0)))).toBe(s)
  })

  it('读查询给每个文件起别名', () => {
    const q = readQuery(paths)
    expect(q).toContain('f0: file(path: "src/data/a.json")')
    expect(q).toContain('f1: file(path: "src/data/b.json")')
  })

  it('数据分支存在：读它的 head 和文件', async () => {
    const { fn } = fakeFetch([
      { body: { data: { repository: { id: 'R', defaultBranchRef: { target: commit('main1', ['x', 'y']) }, ref: { target: commit('d1', ['A', null]) } } } } },
    ])
    const snap = await githubRemote({ token: 't', repo, fetch: fn }).read(paths)
    expect(snap).toEqual({ version: 'd1', texts: { [paths[0]]: 'A', [paths[1]]: null } })
  })

  it('数据分支不存在：从默认分支建出来，用默认分支的内容', async () => {
    const { fn, calls } = fakeFetch([
      { body: { data: { repository: { id: 'R', defaultBranchRef: { target: commit('main1', ['x', 'y']) }, ref: null } } } },
      { body: { data: { createRef: { ref: { name: 'data' } } } } },
    ])
    const snap = await githubRemote({ token: 't', repo, fetch: fn }).read(paths)
    expect(snap.version).toBe('main1')
    expect(snap.texts[paths[0]]).toBe('x')
    expect(calls[1].variables.input).toEqual({ repositoryId: 'R', name: 'refs/heads/data', oid: 'main1' })
  })

  it('写：一个 commit，带期望的 head，内容是 base64', async () => {
    const { fn, calls } = fakeFetch([{ body: { data: { createCommitOnBranch: { commit: { oid: 'd2' } } } } }])
    await githubRemote({ token: 't', repo, fetch: fn, device: 'iPhone' }).write([{ path: paths[0], text: '{}\n' }], 'd1', '批注')
    const input = calls[0].variables.input
    expect(input.expectedHeadOid).toBe('d1')
    expect(input.branch).toEqual({ repositoryNameWithOwner: 'o/r', branchName: 'data' })
    expect(input.message.headline).toBe('sync: 批注 · iPhone')
    expect(input.fileChanges.additions).toEqual([{ path: paths[0], contents: toBase64('{}\n') }])
  })

  it('head 对不上映射成 StaleError，401 映射成 AuthError', async () => {
    const stale = fakeFetch([{ body: { errors: [{ type: 'STALE_DATA', message: 'Expected branch to point to "d0" but it did not.' }] } }])
    await expect(githubRemote({ token: 't', repo, fetch: stale.fn }).write([{ path: 'a', text: 'b' }], 'd0', 'x')).rejects.toBeInstanceOf(StaleError)
    const auth = fakeFetch([{ status: 401, body: { message: 'Bad credentials' } }])
    await expect(githubRemote({ token: 't', repo, fetch: auth.fn }).read(paths)).rejects.toBeInstanceOf(AuthError)
  })
})

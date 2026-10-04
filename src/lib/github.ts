/**
 * 用 GitHub GraphQL API 读写仓库里的数据文件，给 sync.ts 当 Remote。
 *
 * - 读：一次查询拿到数据分支的 head 和所有文件内容，保证是同一个版本的快照。
 * - 写：createCommitOnBranch 一次提交多个文件，带上 expectedHeadOid；
 *   这期间别的设备先提交了，GitHub 会拒绝（STALE_DATA），交给 syncDocs 重读重合并。
 * - 数据分支不存在时，从默认分支（main）建出来，内容就是 main 上那几个文件。
 *
 * 数据放在单独的分支上：手机上每次同步都是一个 commit，不混进 main 的历史，也不会触发部署、
 * 不会让电脑上 git push 被拒。部署时再把数据分支的文件拷进来（见 .github/workflows/deploy.yml）。
 */
import { StaleError, type Remote, type RemoteFile, type RemoteSnapshot } from './sync'

export interface RepoConfig {
  owner: string
  name: string
  /** 存数据的分支 */
  branch: string
}

export const DATA_REPO: RepoConfig = { owner: 'scatyf3', name: 'ai-infra-inferview', branch: 'data' }

export const GRAPHQL_URL = 'https://api.github.com/graphql'

/** token 不对、过期或被吊销 */
export class AuthError extends Error {
  constructor(message = 'GitHub token 无效或已过期') {
    super(message)
    this.name = 'AuthError'
  }
}

export class GitHubError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'GitHubError'
  }
}

type Fetch = (url: string, init: RequestInit) => Promise<Response>

interface GqlError {
  type?: string
  message: string
}

export async function gql<T>(fetchFn: Fetch, token: string, query: string, variables: object, keepalive = false): Promise<T> {
  const body = JSON.stringify({ query, variables })
  let res: Response
  try {
    res = await fetchFn(GRAPHQL_URL, {
      method: 'POST',
      headers: { Authorization: `bearer ${token}`, 'Content-Type': 'application/json' },
      body,
      // 页面切到后台时也尽量把请求发完；浏览器限制 keepalive 的请求体不超过 64KB
      keepalive: keepalive && body.length < 60_000,
    })
  } catch (e) {
    throw new GitHubError(`连不上 GitHub：${(e as Error).message}`)
  }
  if (res.status === 401) throw new AuthError()
  const json = (await res.json().catch(() => null)) as { data?: T; errors?: GqlError[]; message?: string } | null
  if (!res.ok) throw new GitHubError(`GitHub 返回 ${res.status}：${json?.message ?? res.statusText}`)
  if (json?.errors?.length) {
    const stale = json.errors.find((e) => e.type === 'STALE_DATA' || /expected branch to point to/i.test(e.message))
    if (stale) throw new StaleError(stale.message)
    const denied = json.errors.find((e) => e.type === 'FORBIDDEN' || /resource not accessible/i.test(e.message))
    if (denied) throw new AuthError('这个 token 没有写这个仓库的权限（需要 Contents: Read and write）')
    throw new GitHubError(json.errors.map((e) => e.message).join('；'))
  }
  if (!json?.data) throw new GitHubError('GitHub 返回了空结果')
  return json.data
}

// ---------- 编码 ----------

/** UTF-8 文本转 base64（btoa 只认 Latin-1，中文要先编码成字节） */
export function toBase64(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let bin = ''
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(bin)
}

// ---------- 读 ----------

const fileFields = (paths: string[]) =>
  paths.map((p, i) => `f${i}: file(path: ${JSON.stringify(p)}) { object { ... on Blob { text } } }`).join('\n')

/** 一次查询：仓库 id、默认分支和数据分支各自的 head 与文件内容 */
export function readQuery(paths: string[]): string {
  const files = fileFields(paths)
  return `query($owner: String!, $name: String!, $ref: String!) {
  repository(owner: $owner, name: $name) {
    id
    defaultBranchRef { target { ... on Commit { oid
${files}
    } } }
    ref(qualifiedName: $ref) { target { ... on Commit { oid
${files}
    } } }
  }
}`
}

type FileEntry = { object: { text: string | null } | null } | null
type CommitNode = { oid: string } & Record<string, FileEntry | string>

interface ReadData {
  repository: {
    id: string
    defaultBranchRef: { target: CommitNode } | null
    ref: { target: CommitNode } | null
  } | null
}

function textsOf(node: CommitNode, paths: string[]): Record<string, string | null> {
  const out: Record<string, string | null> = {}
  paths.forEach((p, i) => {
    const entry = node[`f${i}`] as FileEntry
    out[p] = entry?.object?.text ?? null
  })
  return out
}

const CREATE_REF = `mutation($input: CreateRefInput!) { createRef(input: $input) { ref { name } } }`

const COMMIT = `mutation($input: CreateCommitOnBranchInput!) {
  createCommitOnBranch(input: $input) { commit { oid } }
}`

export interface GitHubRemoteOptions {
  token: string
  repo?: RepoConfig
  fetch?: Fetch
  /** 写进 commit 信息，比如 iPhone、Android、桌面 */
  device?: string
}

export function githubRemote({ token, repo = DATA_REPO, fetch: fetchFn = (u, i) => fetch(u, i), device }: GitHubRemoteOptions): Remote & {
  /** 页面切到后台时的最后一次同步用：请求带 keepalive */
  keepalive: boolean
} {
  const vars = { owner: repo.owner, name: repo.name, ref: `refs/heads/${repo.branch}` }

  const remote = {
    keepalive: false,

    async read(paths: string[]): Promise<RemoteSnapshot> {
      const data = await gql<ReadData>(fetchFn, token, readQuery(paths), vars, remote.keepalive)
      const r = data.repository
      if (!r) throw new AuthError(`读不到仓库 ${repo.owner}/${repo.name}：token 没有授权这个仓库`)
      if (r.ref) return { texts: textsOf(r.ref.target, paths), version: r.ref.target.oid }

      // 数据分支还没有：从默认分支建出来，内容就是默认分支上的文件
      const base = r.defaultBranchRef?.target
      if (!base) throw new GitHubError('仓库没有默认分支')
      try {
        await gql(fetchFn, token, CREATE_REF, { input: { repositoryId: r.id, name: vars.ref, oid: base.oid } })
      } catch (e) {
        // 另一台设备刚好同时建了：重读一次就好
        if (e instanceof GitHubError && /already exists/i.test(e.message)) return remote.read(paths)
        throw e
      }
      return { texts: textsOf(base, paths), version: base.oid }
    },

    async write(files: RemoteFile[], version: string | null, summary: string): Promise<void> {
      if (!version) throw new StaleError('不知道当前的分支版本')
      const input = {
        branch: { repositoryNameWithOwner: `${repo.owner}/${repo.name}`, branchName: repo.branch },
        message: { headline: `sync: ${summary}${device ? ` · ${device}` : ''}` },
        expectedHeadOid: version,
        fileChanges: { additions: files.map((f) => ({ path: f.path, contents: toBase64(f.text) })) },
      }
      await gql(fetchFn, token, COMMIT, { input }, remote.keepalive)
    },
  }
  return remote
}

export interface TokenInfo {
  login: string
  /** 对数据仓库的权限：WRITE / MAINTAIN / ADMIN 才能同步 */
  permission: string | null
}

/** 保存 token 前先验一下：能不能登录、对数据仓库有没有写权限 */
export async function checkToken(token: string, repo: RepoConfig = DATA_REPO, fetchFn: Fetch = (u, i) => fetch(u, i)): Promise<TokenInfo> {
  const data = await gql<{ viewer: { login: string }; repository: { viewerPermission: string | null } | null }>(
    fetchFn,
    token,
    `query($owner: String!, $name: String!) { viewer { login } repository(owner: $owner, name: $name) { viewerPermission } }`,
    { owner: repo.owner, name: repo.name },
  )
  return { login: data.viewer.login, permission: data.repository?.viewerPermission ?? null }
}

export const canWrite = (p: string | null) => p === 'WRITE' || p === 'MAINTAIN' || p === 'ADMIN'

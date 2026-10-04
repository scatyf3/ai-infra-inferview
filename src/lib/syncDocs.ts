/**
 * 要在设备间同步的几份读者数据：存在仓库的哪个文件、本地缓存用哪个 key、怎么校验 / 合并 / 输出。
 * dev server 的读写接口（docs/.vitepress/dataStore.ts）和页面里的同步（docs/.vitepress/theme/sync.ts）都从这里取。
 */
import {
  isFlags,
  isNotes,
  isProgress,
  mergeFlags,
  mergeNotes,
  mergeProgress,
  serializeFlags,
  serializeNotes,
  serializeProgress,
  type Flags,
  type Notes,
  type Progress,
} from './flashcards'
import { isReaderNotes, mergeReaderNotes, serializeReaderNotes, type ReaderNotes } from './readerNotes'

export interface DocSpec<T> {
  /** 短名，dev server 接口是 /__data/<key> */
  key: string
  /** 仓库里的路径 */
  path: string
  /** 给人看的名字，用在同步的 commit 信息里 */
  label: string
  /** 本地缓存的 localStorage key；沿用以前的 key，老数据能直接接上 */
  localKey: string
  empty: () => T
  validate: (v: unknown) => v is T
  /** b 里较新的条目盖过 a */
  merge: (a: T, b: T) => T
  serialize: (v: T) => string
}

export const readerNotesSpec: DocSpec<ReaderNotes> = {
  key: 'reader-notes',
  path: 'src/data/reader-notes.json',
  label: '批注',
  localKey: 'inferview:reader-notes:v1',
  empty: () => ({}),
  validate: isReaderNotes,
  merge: mergeReaderNotes,
  serialize: serializeReaderNotes,
}

export const progressSpec: DocSpec<Progress> = {
  key: 'flashcard-progress',
  path: 'src/data/flashcard-progress.json',
  label: '闪卡复习',
  localKey: 'inferview:flashcards:v2',
  empty: () => ({}),
  validate: isProgress,
  merge: mergeProgress,
  serialize: serializeProgress,
}

export const cardNotesSpec: DocSpec<Notes> = {
  key: 'flashcard-notes',
  path: 'src/data/flashcard-notes.json',
  label: '闪卡批注',
  localKey: 'inferview:flashcard-notes:v1',
  empty: () => ({}),
  validate: isNotes,
  merge: mergeNotes,
  serialize: serializeNotes,
}

export const flagsSpec: DocSpec<Flags> = {
  key: 'flashcard-flags',
  path: 'src/data/flashcard-flags.json',
  label: '闪卡暂停',
  localKey: 'inferview:flashcard-flags:v1',
  empty: () => ({}),
  validate: isFlags,
  merge: mergeFlags,
  serialize: serializeFlags,
}

/** 全部要同步的数据；同步一轮就是一起读、一起写（GitHub 上合成一个 commit） */
export const SYNC_DOCS: DocSpec<any>[] = [readerNotesSpec, progressSpec, cardNotesSpec, flagsSpec]

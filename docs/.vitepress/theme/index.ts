import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import KnowledgeMap from './components/KnowledgeMap.vue'
import MemoryCalculator from './components/MemoryCalculator.vue'
import ShapeFlow from './components/ShapeFlow.vue'
import ParallelismViz from './components/ParallelismViz.vue'
import PagedKV from './components/PagedKV.vue'
import ReleaseTimeline from './components/ReleaseTimeline.vue'
import ReaderNotes from './components/ReaderNotes.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  // 全站挂载读者划词高亮 / 批注
  Layout: () => h(DefaultTheme.Layout, null, { 'layout-bottom': () => h(ReaderNotes) }),
  enhanceApp({ app }) {
    app.component('KnowledgeMap', KnowledgeMap)
    app.component('MemoryCalculator', MemoryCalculator)
    app.component('ShapeFlow', ShapeFlow)
    app.component('ParallelismViz', ParallelismViz)
    app.component('PagedKV', PagedKV)
    app.component('ReleaseTimeline', ReleaseTimeline)
  },
} satisfies Theme

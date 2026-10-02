import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import MemoryCalculator from './components/MemoryCalculator.vue'
import ShapeFlow from './components/ShapeFlow.vue'
import ParallelismViz from './components/ParallelismViz.vue'
import PagedKV from './components/PagedKV.vue'
import ReleaseTimeline from './components/ReleaseTimeline.vue'
import HandsonProgress from './components/HandsonProgress.vue'
import LeetGPUBoard from './components/LeetGPUBoard.vue'
import LeetGPURoadmap from './components/LeetGPURoadmap.vue'
import StackFigure from './components/StackFigure.vue'
import RoadmapFigure from './components/RoadmapFigure.vue'
import StackPageExtras from './components/StackPageExtras.vue'
import ReaderNotes from './components/ReaderNotes.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  // 全站挂载读者划词高亮 / 批注；/stack/<id> 页额外加上所在层和相关文章（其他页面渲染为空）
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(ReaderNotes),
      'doc-before': () => h(StackPageExtras, { where: 'before' }),
      'doc-after': () => h(StackPageExtras, { where: 'after' }),
    }),
  enhanceApp({ app }) {
    app.component('MemoryCalculator', MemoryCalculator)
    app.component('ShapeFlow', ShapeFlow)
    app.component('ParallelismViz', ParallelismViz)
    app.component('PagedKV', PagedKV)
    app.component('ReleaseTimeline', ReleaseTimeline)
    app.component('HandsonProgress', HandsonProgress)
    app.component('LeetGPUBoard', LeetGPUBoard)
    app.component('LeetGPURoadmap', LeetGPURoadmap)
    app.component('StackFigure', StackFigure)
    app.component('RoadmapFigure', RoadmapFigure)
  },
} satisfies Theme

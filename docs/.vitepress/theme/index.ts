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
import TritonProgramViz from './components/TritonProgramViz.vue'
import Flashcards from './components/Flashcards.vue'
import QuantGranularity from './components/QuantGranularity.vue'
import StackPageExtras from './components/StackPageExtras.vue'
import ReaderNotes from './components/ReaderNotes.vue'
import SyncControl from './components/SyncControl.vue'
import { registerServiceWorker } from './pwa'
import 'katex/dist/katex.min.css'
import './custom.css'

export default {
  extends: DefaultTheme,
  // 全站挂载读者划词高亮 / 批注和导航栏上的同步状态；/stack/<id> 页额外加上所在层和相关文章（其他页面渲染为空）
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(ReaderNotes),
      'nav-bar-content-after': () => h(SyncControl),
      'doc-before': () => h(StackPageExtras, { where: 'before' }),
      'doc-after': () => h(StackPageExtras, { where: 'after' }),
    }),
  enhanceApp({ app }) {
    registerServiceWorker()
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
    app.component('TritonProgramViz', TritonProgramViz)
    app.component('Flashcards', Flashcards)
    app.component('QuantGranularity', QuantGranularity)
  },
} satisfies Theme

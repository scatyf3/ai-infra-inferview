import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import KnowledgeMap from './components/KnowledgeMap.vue'
import MemoryCalculator from './components/MemoryCalculator.vue'
import ShapeFlow from './components/ShapeFlow.vue'
import ParallelismViz from './components/ParallelismViz.vue'
import PagedKV from './components/PagedKV.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('KnowledgeMap', KnowledgeMap)
    app.component('MemoryCalculator', MemoryCalculator)
    app.component('ShapeFlow', ShapeFlow)
    app.component('ParallelismViz', ParallelismViz)
    app.component('PagedKV', PagedKV)
  },
} satisfies Theme

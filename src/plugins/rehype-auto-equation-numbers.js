// lib/plugins/rehype-auto-equation-numbers.js
import { visit } from 'unist-util-visit';

export default function autoEquationNumbers() {
  return (tree) => {
    let eqCount = 0;
    visit(tree, 'element', (node) => {
      if (node.tagName === 'span' && node.properties?.className?.includes('katex-display')) {
        eqCount++;
        const label = node.properties.id || `eq:auto-${eqCount}`;
        node.children.push({
          type: 'element',
          tagName: 'span',
          properties: { class: 'equation-number' },
          children: [{ type: 'text', value: `(${eqCount})` }]
        });
        node.properties.id = label;
      }
    });
  };
}

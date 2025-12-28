/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Python for Quants',
      items: [
        'python-for-quants/getting-started',
        'python-for-quants/numpy-pandas',
        'python-for-quants/data-visualization',
      ],
    },
    {
      type: 'category',
      label: 'Financial Markets',
      items: [
        'financial-markets/market-basics',
        'financial-markets/trading-strategies',
        'financial-markets/risk-management',
      ],
    },
    {
      type: 'category',
      label: 'Quantitative Analysis',
      items: [
        'quantitative-analysis/statistical-methods',
        'quantitative-analysis/backtesting',
        'quantitative-analysis/portfolio-optimization',
      ],
    },
    {
      type: 'category',
      label: 'Tools & Libraries',
      items: [
        'tools-libraries/data-sources',
        'tools-libraries/backtesting-frameworks',
        'tools-libraries/machine-learning',
      ],
    },
  ],
};

module.exports = sidebars;

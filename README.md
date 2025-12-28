# QuantClub Resource Wiki

A comprehensive learning resource for quantitative development and algorithmic trading, built with [Docusaurus](https://docusaurus.io/).

🌐 **Live Site**: `https://YOUR_GITHUB_USERNAME.github.io/ResourceWiki/`

## 📚 What's Inside

This wiki covers:

- **Python for Quants**: Master Python programming for financial applications
- **Financial Markets**: Understanding market structures, instruments, and trading
- **Quantitative Analysis**: Statistical methods, backtesting, and portfolio optimization
- **Tools & Libraries**: Essential frameworks and data sources for quant development

## 🚀 Running Locally

### Prerequisites

Make sure you have installed:
- **Node.js** version 18.0 or higher ([Download here](https://nodejs.org/))
- **npm** (comes with Node.js) or **yarn**

Check your versions:
```bash
node --version  # Should be v18.0.0 or higher
npm --version   # Should be 9.0.0 or higher
```

### Step 1: Clone or Navigate to Project

```bash
# If cloning from GitHub
git clone https://github.com/YOUR_GITHUB_USERNAME/ResourceWiki.git
cd ResourceWiki

# OR if you already have the project
cd ResourceWiki
```

### Step 2: Install Dependencies

```bash
npm install
```

This will install all required packages (may take 1-2 minutes).

### Step 3: Start Development Server

```bash
npm start
```

The site will automatically open in your browser at `http://localhost:3000`

**That's it!** 🎉 The development server will:
- Hot-reload when you save changes
- Show any errors in the console
- Automatically refresh your browser

### Common Commands

```bash
# Start development server
npm start

# Build for production
npm run build

# Serve production build locally
npm run serve

# Clear cache (if you have issues)
npm run clear
```

### Troubleshooting

**Port 3000 already in use?**
```bash
# Kill the process or the server will use port 3001 automatically
```

**Build errors?**
```bash
# Clear cache and reinstall
npm run clear
rm -rf node_modules package-lock.json
npm install
```

**Changes not showing?**
- Save the file (Ctrl+S / Cmd+S)
- Check the terminal for errors
- Try refreshing the browser (F5)

### Build

```bash
# Build the static website
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## 📖 Documentation Structure

```
docs/
├── intro.md                          # Introduction
├── python-for-quants/               # Python tutorials
│   ├── getting-started.md
│   ├── numpy-pandas.md
│   └── data-visualization.md
├── financial-markets/               # Market fundamentals
│   ├── market-basics.md
│   ├── trading-strategies.md
│   └── risk-management.md
├── quantitative-analysis/           # Quant methods
│   ├── statistical-methods.md
│   ├── backtesting.md
│   └── portfolio-optimization.md
└── tools-libraries/                 # Tools and frameworks
    ├── data-sources.md
    ├── backtesting-frameworks.md
    └── machine-learning.md
```

## 🌐 Deploy to GitHub Pages

### Step 1: Update Configuration

Edit `docusaurus.config.js`:

```javascript
module.exports = {
  // ...
  url: 'https://YOUR_GITHUB_USERNAME.github.io',
  baseUrl: '/ResourceWiki/',
  organizationName: 'YOUR_GITHUB_USERNAME',
  projectName: 'ResourceWiki',
  // ...
};
```

### Step 2: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select:
   - **Source**: GitHub Actions

### Step 3: Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

The GitHub Actions workflow will automatically build and deploy your site!

### Step 4: Access Your Site

Your site will be available at: `https://YOUR_GITHUB_USERNAME.github.io/ResourceWiki/`

## 🔧 Customization

### Changing Colors

Edit `src/css/custom.css`:

```css
:root {
  --ifm-color-primary: #2e8555;
  --ifm-color-primary-dark: #29784c;
  /* ... */
}
```

### Adding New Pages

1. Create a new `.md` file in the `docs/` directory
2. Add frontmatter:

```markdown
---
sidebar_position: 1
---

# Your Page Title

Your content here...
```

3. The page will automatically appear in the sidebar

### Adding Blog Posts

Create a new file in `blog/`:

```markdown
---
slug: your-post-slug
title: Your Post Title
authors: [quantclub]
tags: [tag1, tag2]
---

Your blog post content...
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Useful Links

- [Docusaurus Documentation](https://docusaurus.io/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Markdown Guide](https://www.markdownguide.org/)

## 📧 Contact

- GitHub: [@YOUR_GITHUB_USERNAME](https://github.com/YOUR_GITHUB_USERNAME)
- Project Link: [https://github.com/YOUR_GITHUB_USERNAME/ResourceWiki](https://github.com/YOUR_GITHUB_USERNAME/ResourceWiki)

## 🎓 Learning Resources

For additional learning materials, check out:

- [Python Documentation](https://docs.python.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Quantitative Finance on Stack Exchange](https://quant.stackexchange.com/)

---

Built with ❤️ by QuantClub BITSP

**Happy Learning! 🚀**

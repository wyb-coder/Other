# To-Do PWA 移动端版本

这是 To-Do 应用的 PWA (Progressive Web App) 版本，专为移动端优化。

## 📁 文件结构

```
Mobile To-Do/
├── index.html      # 主应用文件 (含PWA支持)
├── manifest.json   # PWA配置文件
├── sw.js           # Service Worker (离线缓存)
├── icons/          # 应用图标
│   ├── icon-192x192.png
│   └── icon-512x512.png
└── README.md       # 本说明文件
```

## 🚀 部署方式

### 方式一：本地测试

```bash
# 使用 Python
python -m http.server 3000

# 或使用 Node.js
npx serve . -l 3000
```

然后在浏览器访问 `http://localhost:3000`

### 方式二：部署到静态托管 (推荐)

1. **GitHub Pages** (免费)

   - 创建 GitHub 仓库
   - 上传此文件夹内容
   - 在 Settings → Pages 中启用

2. **Vercel** (免费)

   - 访问 https://vercel.com
   - 拖拽上传此文件夹

3. **Netlify** (免费)
   - 访问 https://netlify.com
   - 拖拽上传此文件夹

## 📱 安装到手机主屏幕

### iOS (Safari)

1. 用 Safari 打开部署后的网址
2. 点击分享按钮 (方框+箭头)
3. 选择"添加到主屏幕"
4. 点击"添加"

### Android (Chrome)

1. 用 Chrome 打开部署后的网址
2. 点击右上角三个点菜单
3. 选择"添加到主屏幕"或"安装应用"
4. 确认安装

## ✨ PWA 特性

- **离线可用**: 首次加载后，无网络也能使用
- **原生体验**: 全屏显示，无浏览器地址栏
- **快速启动**: 安装后从主屏幕秒开
- **自动更新**: 有新版本时自动更新缓存

## 📝 数据存储

- 数据存储在浏览器 LocalStorage 中
- 建议定期使用"数据管理"功能导出备份
- 可开启"每月自动备份"功能

## ⚠️ 注意事项

1. **PWA 需要 HTTPS**: 部署时必须使用 HTTPS 协议
2. **数据迁移**: PC 端和移动端数据不共享，需手动导入导出
3. **浏览器支持**: 推荐使用最新版 Chrome/Safari

## 🔄 从 PC 端迁移数据

1. 在 PC 端打开 To-Do 应用
2. 点击"数据管理" → "立即手动导出"
3. 将 JSON 文件传输到手机
4. 在移动端 PWA 中点击"数据管理" → "从备份恢复数据"

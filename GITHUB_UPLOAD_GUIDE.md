# GitHub上传指南

## 上传前准备

### 1. 确认关键文件已准备

确保以下文件已创建并正确配置：
- ✅ `README.md` - 项目说明文档
- ✅ `requirements.txt` - Python依赖包列表
- ✅ `.gitignore` - Git忽略文件
- ✅ `setup.py` - 项目安装配置
- ✅ `data_config.json` - 数据集配置

### 2. 核心代码文件

以下文件是必须上传的核心代码：
- ✅ `run_ecoli.py` - 主训练脚本
- ✅ `model.py` - 深度学习模型定义
- ✅ `data_loader.py` - 数据加载和预处理
- ✅ `load_data.py` - 数据集加载工具
- ✅ `cross_validate_ecoli.py` - 交叉验证脚本
- ✅ `visualize_model_components.py` - 模型可视化

## GitHub上传步骤

### 步骤1：创建GitHub仓库

1. 登录GitHub
2. 点击右上角 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - Repository name: `gene-regulation-prediction`
   - Description: `Deep learning model for gene regulation network prediction`
   - 选择 Public 或 Private
   - 不要勾选 "Initialize this repository with a README"
4. 点击 "Create repository"

### 步骤2：初始化本地Git仓库

```bash
# 在项目目录下执行
git init
git add .
git commit -m "Initial commit: Gene regulation prediction model"
```

### 步骤3：连接远程仓库

```bash
# 替换为你的GitHub用户名和仓库名
git remote add origin https://github.com/yourusername/gene-regulation-prediction.git
git branch -M main
git push -u origin main
```

## 文件上传策略

### 必须上传的文件

```
📁 gene-regulation-prediction/
├── 📄 README.md                    # 项目说明
├── 📄 requirements.txt             # 依赖包列表
├── 📄 setup.py                     # 安装配置
├── 📄 .gitignore                   # Git忽略文件
├── 📄 data_config.json             # 数据配置
├── 📄 run_ecoli.py                 # 主训练脚本
├── 📄 model.py                     # 模型定义
├── 📄 data_loader.py               # 数据加载
├── 📄 load_data.py                 # 数据集工具
├── 📄 cross_validate_ecoli.py      # 交叉验证
└── 📄 visualize_model_components.py # 可视化工具
```

### 不会上传的文件（被.gitignore排除）

- ❌ `*.pth` - 模型文件（通常很大）
- ❌ `*.log` - 日志文件
- ❌ `results_*/` - 结果目录
- ❌ `models_*/` - 模型目录
- ❌ `__pycache__/` - Python缓存
- ❌ `*.bak`, `*_backup_*` - 备份文件
- ❌ `*.png`, `*.jpg` - 图片文件

## 验证上传

### 检查上传的文件

1. 访问你的GitHub仓库页面
2. 确认以下文件已上传：
   - README.md
   - requirements.txt
   - setup.py
   - .gitignore
   - 所有核心Python文件

### 验证.gitignore效果

确认以下文件**没有**被上传：
- 任何.pth文件
- 日志文件
- 结果目录
- 缓存目录

## 后续维护

### 更新代码

```bash
# 修改代码后
git add .
git commit -m "Update: [描述你的更改]"
git push origin main
```

### 添加新功能

```bash
# 创建新分支
git checkout -b feature/new-feature
# 开发完成后
git add .
git commit -m "Add: [新功能描述]"
git push origin feature/new-feature
# 创建Pull Request
```

## 常见问题

### Q: 模型文件太大无法上传？

A: 模型文件（.pth）已被.gitignore排除。如果需要分享模型，建议：
- 使用Git LFS（Large File Storage）
- 上传到云存储（如Google Drive、OneDrive）
- 在README中提供模型下载链接

### Q: 如何让其他人使用我的代码？

A: 在README.md中提供：
- 详细的安装步骤
- 使用示例
- 数据格式说明
- 常见问题解答

### Q: 如何添加许可证？

A: 创建LICENSE文件：
```bash
# 创建MIT许可证
echo "MIT License" > LICENSE
git add LICENSE
git commit -m "Add: MIT License"
git push origin main
```

## 最佳实践

1. **保持README更新**：每次重要更新后更新README
2. **使用语义化提交**：使用清晰的提交信息
3. **添加标签**：为重要版本添加Git标签
4. **维护依赖**：定期更新requirements.txt
5. **文档化**：为复杂功能添加详细注释

## 仓库设置建议

### 启用功能

1. **Issues**：用于bug报告和功能请求
2. **Wiki**：用于详细文档
3. **Projects**：用于项目管理
4. **Actions**：用于CI/CD（可选）

### 分支保护

1. 在仓库设置中启用分支保护
2. 要求Pull Request审查
3. 要求状态检查通过

---

**注意**：上传前请确保没有敏感信息（如API密钥、个人数据等）在代码中。 
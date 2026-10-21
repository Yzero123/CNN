# 贡献指南

感谢你愿意为这个项目出力。这份文档说明如何把改动顺利合进来。

## 环境准备

```bash
git clone https://github.com/Yzero123/CNN.git
cd CNN

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

需要 Python 3.9 及以上。TensorFlow 体积较大，建议使用独立虚拟环境。

## 本地验证

提交前请确保下面三条命令都通过：

```bash
# 语法检查
python -m py_compile 3cl_train.py predict.py

# 用仓库自带模型做一次预测
python predict.py img_test/dog/1001.jpg

# 快速训练（1 轮，仅验证流程能跑通）
python 3cl_train.py --epochs 1 --model-out /tmp/smoke.h5
```

## 代码约定

- 遵循 [PEP 8](https://peps.python.org/pep-0008/)，4 空格缩进，UTF-8 编码。
- 脚本面向初学者，**关键步骤保留中文注释**，说明这一行在做什么。
- 新增命令行参数一律走 `argparse`，并在 `--help` 里写清默认值。
- 不要提交新的模型权重文件（`.h5` 通常在几十 MB），除非明确讨论过。
- 不要提交 `__pycache__/`、虚拟环境或本地训练输出。

## 关于数据集

- 训练集放在 `img_data/<类别名>/`，测试集放在 `img_test/<类别名>/`。
- 类别名就是模型输出的标签顺序，**按字母序排列**。新增类别后，脚本会自动从目录结构推导标签，无需改代码。
- 如果你发现标签错误（图片内容与类别不符），请直接开 Issue 并说明文件路径——这类问题会显著拉低准确率。

## 提交信息

用祈使句描述改动本身：

```
feat: 训练脚本支持 --data-dir 与 --epochs 参数
fix: 预测脚本的类别名改为从数据集目录推导
docs: 补充实测准确率与数据集说明
```

## Pull Request

1. 从 `main` 切出特性分支：`git checkout -b feat/your-feature`
2. 提交并推送：`git push -u origin feat/your-feature`
3. 开 PR，描述里写清**改了什么**和**怎么验证的**。
4. 如果改动了模型或训练流程，请附上新的实测准确率（跑一遍 `img_test/` 全量）。

## 报告 Bug

请用 [Bug 报告模板](.github/ISSUE_TEMPLATE/bug_report.md)，包含：

- 操作系统与 Python 版本
- TensorFlow 版本
- 复现步骤
- 完整报错信息

## 需要帮助？

开一个 Issue 描述你的想法，附上使用场景。

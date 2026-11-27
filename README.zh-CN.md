# RoboFailAtlas

本仓库包含奖励模型训练与评测、策略学习适配代码和项目网站。
安装与运行入口见 [README](README.md)。

- `src/`、`scripts/`：模型、数据接口、训练与评测入口。
- `configs/`：共享参数、三类 RM 配置、五个任务及消融配置。
- `integrations/`：PPO 与失败指导 SFT 适配。
- `documentation/`：数据格式、指标、运行环境和配置说明。
- `docs/`：静态网站、论文图片、演示视频与轨迹案例。
- `examples/`、`tests/`：样例、保存预测的指标核对与 CPU 检查。

配置精简保留了原模板参数。历史 PPO 模板预算与后续论文设置的差异见
[配置说明](documentation/CONFIGURATION.md)，不把 CPU 检查等同于 GPU 完整复现。

保密标注管线、完整 prompt、任务评分实现、内部标注服务、模型权重及外部
训练数据均未包含。项目许可证尚未确定，第三方许可文件保留。

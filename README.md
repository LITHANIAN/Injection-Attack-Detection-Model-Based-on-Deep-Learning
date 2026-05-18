# Injection-Attack-Detection-Model-Based-on-Deep-Learning
本项目是基于深度学习的注入攻击流量检测方法，基于PyTorch框架构建混合神经网络模型，实现对HTTP请求中SQL注入、命令注入等攻击载荷的端到端智能检测。该方法支持HTTP请求分词向量化、多尺度CNN特征提取、混淆编码绕过检测、BiLSTM上下文建模、自注意力权重可视化及多阈值决策，可用于Web应用防火墙、云端流量安全审计、安全运营中心告警等场景。
环境要求
Python 3.8 及以上
安装依赖
pip install -r requirements.txt
日志说明
model_train.log：模型训练日志
ablation_model_train.log：消融模型训练日志
项目声明
项目名称：基于深度学习的注入攻击流量检测研究
项目作者：Li Yeyang
作者单位：暨南大学网络空间安全学院
开发语言：Python
框架：PyTorch
核心技术：HTTP 请求分词与向量化、多尺度 CNN 特征提取、混淆特征增强分支、双向 LSTM 序列建模、自注意力机制

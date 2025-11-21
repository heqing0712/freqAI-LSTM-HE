![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

# 使用 LSTM 的动态加权与聚合评分交易模型

一个基于 [freqtrade](https://github.com/freqtrade/freqtrade) 的加密货币交易机器人之 [FreqAI](https://www.freqtrade.io/en/stable/freqai/) 模块的回归模型与交易策略。

⚠️ **由于 Freqtrade 最新版本（> 2024.02）开始出现问题，本模型（以及潜在的其他模型）已迁移至 PyTorch。PyTorch 在各平台上拥有更好的 GPU 支持，并且由于无需修改 freqAI 核心（可能只需把最小时间框从 5 分钟提高），能更快迭代开发。** ⚠️

## 概述

本项目旨在构建一个利用动态加权与聚合评分系统来做出更明智交易决策的模型。模型最初基于 TensorFlow 与 Keras API 实现，现已迁移至 PyTorch，以利用其更好的跨平台 GPU 支持与更高的开发效率。

## 快速开始

- 克隆代码仓库：

```shell
git clone https://github.com/Netanelshoshan/freqAI-LSTM.git
```
- 将文件复制到 freqtrade 目录：

```shell
cp torch/BasePyTorchModel.py <freqtrade dir>/freqtrade/freqai/base_models/
cp torch/PyTorchLSTMModel.py <freqtrade dir >/freqtrade/freqai/torch/
cp torch/PyTorchModelTrainer.py <freqtrade dir>/freqtrade/freqai/torch/
cp torch/PyTorchLSTMRegressor.py <freqtrade dir>/user_data/freqaimodels/
cp config-example.json <freqtrade dir>/user_data/config.json
cp ExampleLSTMStrategy.py <freqtrade dir>/user_data/strategies/
```
- 下载数据：

```shell
freqtrade download-data -c user_data/config-torch.json --timerange 20230101-20240529 --timeframe 15m 30m 1h 2h 4h 8h 1d --erase
```
- 编辑 `freqtrade/configuration/config_validation.py`：

```python
...
def _validate_freqai_include_timeframes()
...
    if freqai_enabled:
        main_tf = conf.get('timeframe', '5m') -> 改为 '1h' 或你选择的更小时间框架（最小）
```
- 确保在这些更改后以可编辑模式安装包：

```shell
pip install -e .
```

- 运行回测：

```shell
freqtrade backtesting -c user_data/config-torch.json --breakdown day week month --timerange 20240301-20240401
```

## 使用 Docker 快速开始

- 克隆代码仓库：

```shell
git clone https://github.com/Netanelshoshan/freqAI-LSTM.git
```
- 本地构建镜像：

```shell
cd freqAI-LSTM
docker build -f torch/Dockerfile -t freqai .
```
- 下载数据并运行回测：

```
docker run -v ./data:/freqtrade/user_data/data -it freqai download-data -c user_data/config-torch.json --timerange 20230101-20240529 --timeframe 15m 30m 1h 2h 4h 8h 1d --erase

docker run -v ./data:/freqtrade/user_data/data -it freqai backtesting -c user_data/config-torch.json --breakdown day week month --timerange 20240301-20240401
```

## 模型架构

模型核心为长短期记忆网络（LSTM），属于循环神经网络的一种，擅长处理序列数据并捕捉长期依赖。

LSTM 模型（`PyTorchLSTMModel`）的结构如下：

- 输入数据经过若干层 LSTM（层数由 `num_lstm_layers` 参数配置）。每个 LSTM 层后接批归一化（BatchNorm）与 Dropout 进行正则化。
- 最后一层 LSTM 的输出传入一个 ReLU 激活的全连接层。
- 应用 Alpha Dropout 进行进一步正则化。
- 最后通过另一全连接层给出最终预测。

模型的超参数（如 LSTM 层数、隐藏维度、Dropout 比例等）可在配置文件 `model_training_parameters` 的 `model_kwargs` 中便捷地设置。

示例配置：

```json
"model_training_parameters": {
  "learning_rate": 3e-3,
  "trainer_kwargs": {
    "n_steps": null,
    "batch_size": 32,
    "n_epochs": 10
  },
  "model_kwargs": {
    "num_lstm_layers": 3,
    "hidden_dim": 128,
    "dropout_percent": 0.4,
    "window_size": 5
  }
}
```

参数说明：

- `learning_rate`：优化器学习率，控制每次权重更新的步长。
- `trainer_kwargs`：传递给 `PyTorchLSTMTrainer`（位于 `PyTorchModelTrainer`）的参数。
  - `n_steps`：训练迭代次数；为 `null` 时改用 `n_epochs`。
  - `batch_size`：每次梯度更新的样本数。
  - `n_epochs`：遍历数据集的轮次。
- `model_kwargs`：传递给 `PyTorchLSTMModel` 的参数。
  - `num_lstm_layers`：LSTM 层数。
  - `hidden_dim`：每层 LSTM 的隐藏单元维度。
  - `dropout_percent`：Dropout 比例，用于防止过拟合。
  - `window_size`：预测时回看时间步（或数据点）数量。

## 策略

该策略的核心是从多个角度审视市场，做出更聪明的交易决策。它像一个专家团队，分别关注市场的不同方面，并将见解汇总后再做决策。

工作流程：

- 指标：计算一系列技术指标，作为观察市场的不同“镜头”，帮助识别趋势、动量、波动性等关键特征。
- 归一化：为保证指标可比性，使用 z-score 进行标准化，以便合理加权。
- 动态加权：根据市场状态动态调整不同指标的重要性。
- 聚合评分：将所有标准化后的指标组合成单一分数，代表总体市场情绪，相当于专家投票达成共识。
- 市场状态过滤：考虑当前市场处于多头、空头或中性，如同看天气再决定着装。🌞🌧️
- 波动性调整：根据市场波动性调整目标分数；在剧烈波动时谨慎，平稳时更积极。
- 最终目标分数：将上述因素综合为最终目标分数，为 LSTM 学习提供清晰聚焦的信号。
- 入场与出场信号：依据预测目标分数与阈值确定交易的入场和出场。

## 为何有效

- 多因子目标分数能同时考虑市场的多个侧面，使决策更稳健、信息更充分。
- 通过降噪并聚焦关键信息，目标分数帮助 LSTM 从更干净、意义更强的信号中学习。
- 动态加权与市场状态过滤使策略能适应变化的市场环境，让策略能“思考”并调整。

```python
# 步骤 0：计算新增指标
dataframe['ma'] = ta.SMA(dataframe, timeperiod=10)
dataframe['roc'] = ta.ROC(dataframe, timeperiod=2)
dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'], slowperiod=12,
                                                                            fastperiod=26)
dataframe['momentum'] = ta.MOM(dataframe, timeperiod=4)
dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)
bollinger = ta.BBANDS(dataframe, timeperiod=20)
dataframe['bb_upperband'] = bollinger['upperband']
dataframe['bb_middleband'] = bollinger['middleband']
dataframe['bb_lowerband'] = bollinger['lowerband']
dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
dataframe['obv'] = ta.OBV(dataframe)

# 步骤 1：标准化指标
# 目的：让各指标可比并可分配权重
# 方法：计算 z-score（值减去滚动均值，再除以滚动标准差），得到均值为 0、标准差为 1 的标准化值
dataframe['normalized_stoch'] = (dataframe['stoch'] - dataframe['stoch'].rolling(window=14).mean()) / dataframe['stoch'].rolling(window=14).std()
dataframe['normalized_atr'] = (dataframe['atr'] - dataframe['atr'].rolling(window=14).mean()) / dataframe['atr'].rolling(window=14).std()
dataframe['normalized_obv'] = (dataframe['obv'] - dataframe['obv'].rolling(window=14).mean()) / dataframe['obv'].rolling(window=14).std()
dataframe['normalized_ma'] = (dataframe['close'] - dataframe['close'].rolling(window=10).mean()) / dataframe['close'].rolling(window=10).std()
dataframe['normalized_macd'] = (dataframe['macd'] - dataframe['macd'].rolling(window=26).mean()) / dataframe['macd'].rolling(window=26).std()
dataframe['normalized_roc'] = (dataframe['roc'] - dataframe['roc'].rolling(window=2).mean()) / dataframe['roc'].rolling(window=2).std()
dataframe['normalized_momentum'] = (dataframe['momentum'] - dataframe['momentum'].rolling(window=4).mean()) / dataframe['momentum'].rolling(window=4).std()
dataframe['normalized_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(window=10).mean()) / dataframe['rsi'].rolling(window=10).std()
dataframe['normalized_bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).mean() / (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).std()
dataframe['normalized_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(window=20).mean()) / dataframe['cci'].rolling(window=20).std()

# 动态权重（示例：强趋势时提高动量权重）
trend_strength = abs(dataframe['ma'] - dataframe['close'])

# 通过趋势强度的滚动均值与标准差来判断强趋势
# 阈值设为均值以上 1.5 倍标准差，可按需调整
strong_trend_threshold = trend_strength.rolling(window=14).mean() + 1.5 * trend_strength.rolling(window=14).std()

# 若为强趋势，则提高动量权重
is_strong_trend = trend_strength > strong_trend_threshold

# 将动态权重赋值到数据框
dataframe['w_momentum'] = np.where(is_strong_trend, self.w3.value * 1.5, self.w3.value)

# 步骤 2：计算聚合评分 S
w = [self.w0.value, self.w1.value, self.w2.value, self.w3.value, self.w4.value, self.w5.value, self.w6.value, self.w7.value, self.w8.value]

dataframe['S'] = w[0] * dataframe['normalized_ma'] + w[1] * dataframe['normalized_macd'] + w[2] * dataframe['normalized_roc'] + w[3] * dataframe['normalized_rsi'] + w[4] * dataframe['normalized_bb_width'] + w[5] * dataframe['normalized_cci'] + dataframe['w_momentum'] * dataframe['normalized_momentum'] + self.w8.value * dataframe['normalized_stoch'] + self.w7.value * dataframe['normalized_atr'] + self.w6.value * dataframe['normalized_obv']

# 步骤 3：市场状态过滤 R
# 说明：若价格高于上轨，R 赋值 1；若价格低于下轨，R 赋值 -1；否则为 0
dataframe['R'] = 0
dataframe.loc[(dataframe['close'] > dataframe['bb_middleband']) & (dataframe['close'] > dataframe['bb_upperband']), 'R'] = 1
dataframe.loc[(dataframe['close'] < dataframe['bb_middleband']) & (dataframe['close'] < dataframe['bb_lowerband']), 'R'] = -1

# 基于长期均线的额外市场状态过滤
dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)
dataframe['R2'] = np.where(dataframe['close'] > dataframe['ma_100'], 1, -1)

# 步骤 4：波动性调整 V
# 说明：计算布林带宽度，设定 V 为其倒数；带宽大表示更高波动性
bb_width = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
dataframe['V'] = 1 / bb_width

# 另一种基于 ATR 的波动性调整
dataframe['V2'] = 1 / dataframe['atr']

# 计算最终目标分数 T，综合以上所有因素
dataframe['T'] = dataframe['S'] * dataframe['R'] * dataframe['V'] * dataframe['R2'] * dataframe['V2']

# 将目标分数 T 指定给 AI 目标列
dataframe['&-target'] = dataframe['T']
```

## 综合说明

通过计算并标准化指标、应用动态加权、考虑市场状态、进行波动性调整并采用多因子目标分数，该策略为 LSTM 模型提供了全面且高效的学习信号。

这是技术分析、适应性与深度学习的强力组合，旨在有效把握市场并实现更优的交易表现。

## 挑战与未来改进

- 过拟合：通过 Dropout、正则化、调整层数与神经元数量、调节训练轮次等方式缓解。
- 噪声交易：通过阈值与权重过滤噪声，或引入不相似度度量避免受噪声干扰。
- 在适当的超参数与硬件（如 M1 Max / RTX3070）条件下，模型在 120 天的小数据集上使用精简配置进行回测，准确率可超过 90.0%，同时尽力避免过拟合。

![](https://private-user-images.githubusercontent.com/45298885/335341423-b54c065b-0429-485b-9c70-2184f12692cd.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTczMzExNjYsIm5iZiI6MTcxNzMzMDg2NiwicGF0aCI6Ii80NTI5ODg4NS8zMzUzNDE0MjMtYjU0YzA2NWItMDQyOS00ODViLTljNzAtMjE4NGYxMjY5MmNkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MDIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjAyVDEyMjEwNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTdjMzg0OTNjZGRjOTk0N2M4MzE5OTgyZjhkMTM2ZDViYWVjZTY5MjRjZWNmZjY5ODI0YTZmMjQxM2FhN2Q5MTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.NFmDQ1xFgWZvejd9lVsyI_afSoYx3dOY9rjUd0ASC2g)

两个交易对的回测结果（采用改进后的 PyTorch 模型）。

![](https://private-user-images.githubusercontent.com/45298885/335342184-3b27d994-3bc3-4ea1-ba68-e252a0a03aa2.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTczMzExNjYsIm5iZiI6MTcxNzMzMDg2NiwicGF0aCI6Ii80NTI5ODg4NS8zMzUzNDIxODQtM2IyN2Q5OTQtM2JjMy00ZWExLWJhNjgtZTI1MmEwYTAzYWEyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MDIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjAyVDEyMjEwNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc0NGQyZDFlNTIyZjkzMThhM2M4MjczY2ExOTRkOTg3NGMxMWM5YTJlODFhMjU3MWI3ZjVlMWRlODZlMTk4M2UmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.tx5qNJawxJYRYNrttHh4Vf_5vK_qle019s7TOBMblOI)

2024 年 3 月两组交易对的日收益。模型相对严格，不会产生过多信号。

## 贡献

欢迎贡献！如发现问题或有改进建议，请在 [GitHub 仓库](https://github.com/netanelshoshan/freqAI-LSTM) 提交 Issue 或发起 Pull Request。



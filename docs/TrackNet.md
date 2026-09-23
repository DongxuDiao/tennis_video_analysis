# TrackNet 三帧检测

在新分析或工作台重新分析中选择 TrackNet 三帧时序。球使用 TrackNet，人物和球拍仍使用 YOLO。必须重新分析，旧结果不会改变。

兼容权重：yastrebksv/TrackNet 的 BallTrackerNet 原始 state_dict（256 个热图强度类别），不是 TrackNetV2/V3/V4 任意权重。来源 https://github.com/yastrebksv/TrackNet ，其 README 链接 https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view 。预训练文件位于 data/models/tracknet_tennis.pt（TENNIS_DATA_DIR 可改变数据根目录）。仅用 weights_only=True 加载，不加载任意 pickle 模型对象。来源的代码、权重许可需单独核对后再分发或商用。

输入：BGR，640×360，当前/前一帧/前两帧，除以255。每帧分析，忽略球检测跳帧设置。开头两帧只预热，不伪造观测；时间倒退或间隔超过150ms时重置缓存。输出定位回映到旋转归一后的原视频坐标。热图阈值0.5，连通域中心生成候选并进入原卡尔曼关联；框是热图支持范围，不表示球的真实直径。confidence 是热图强度而非校准概率。不会混入 YOLO 球结果。人物/球拍按原步进检测。

MPS 自动策略失败会重试 CPU，显式 MPS 策略失败会报错。缺失或不兼容权重报错，不静默回退。性能记录包含 tracknet_model_hash；每帧候选 source=tracknet。原 YOLO 模式保留以供对比。输入固定640×360，YOLO的imgsz/置信度设置不控制TrackNet。

验证：用相同视频、相同片段分别建立两个分析，使用工作台对比及人工标注评估。关注误检、漏检、定位误差、断轨以及速度有效窗口。预训练数据是比赛转播，手机练习视频仍可能需要微调；时序模型不保证解决遮挡、低分辨率或多球身份混淆。

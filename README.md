# ONN

## 目录介绍
- code_tf: 训练和测试代码
- demo: 可视化展示代码
    - bram_test.py: 一直写入同一张图片到目标地址
    - tcp: tcp传输图像至pc

## 运行

### 使用数据集写入pl侧
```bash
bash demo/bram_test.sh
```

### 使用上位机显示，zynq中软件进行识别或者写入pl侧；或者用上位机显示+运行
1. 确保zynq和上位机在同一个局域网下。  
2. 打开`demo/demo.py`，修改`83`行左右的address字段的`ip地址`地址为上位机的`ip地址`。  
3. 打开`tcp/recv_pc.py`，修改`65`行左右的address字段的`ip地址`地址为上位机的`ip地址`。  
4. 在上位机中运行命令(需要opencv、numpy、matplotlib)：
    ```bash
    python tcp/recv_pc.py
    ```
5. 在`demo/demo.sh`选择**对应功能的代码运行**
   ```bash
   bash demo/demo.sh
   ```
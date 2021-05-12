## 使用说明

移动到`onn-digital`目录下：
    ```cd ~/project/onn-digital```

运行`gen_bin.sh`文件（需要在`onn-digital`目录下）：
    ```bash code_tf/gen_bin.sh```

## 参数说明

`gen_bin.sh`下共有4个参数，可根据情况进行修改，这里依次对其进行说明：

1. model_path：模型存储路径，一般不需要修改
2. save_bin_num：需要保存前多少张正确的样本。*(设置为0或注释这行，都表示不需要保存样本)*
3. bin_name：想要查看的某个bin文件中间结果。该bin文件需要存储在`{model_path}/bin/`目录下。*(注释该行表示不需要查看任何中间结果)*
4. save_params：是否需要存储参数为bin文件。*(注释该行表示不需要存储参数)*

## 注意

sh文件中注释的代码行需要移动到非注释代码行的下方，否则无法正常执行。
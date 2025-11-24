## 显示当前目录文件

```shell
ls 

# 以列表形式显示
ls -lh

# 显示所有文件（包括隐藏文件）
ls -lah
```

##  切换目录

```shell
cd  <目标路径>

cd ./test1_dir
```

## 复制命令

```shell
# 复制文件
cp <源文件> <目标文件>

cp test1.txt test2.txt

# 复制目录  将dir1整个文件夹复制为dir2
cp -r ./dir1 ./dir2
```

## 移动命令

```shell
# 移动文件/剪切/重命名
mv <源文件> <目标文件>
mv test1.txt text2.txt
mv test1.txt ./test2/test2.txt

# 移动某个文件夹
mv ./test1_dir ./test2_dir
```

## 删除命令

```shell
# !!与windows不同，没有回收站，删除后无法恢复，谨慎操作
# 删除单个文件 
rm <需要删除的文件>
rm test1.txt

# 删除整个文件夹
rm -rf <需要删除的目录>
rm -rf ./test_dir
```

## SSH连接

```shell
ssh <username>@<IP Address> # ssh服务默认端口号为22

ssh user1@192.168.0.1 # 创建连接后根据提示输入密码进行验证

# 若ssh服务端口不为22，需在连接时指定对应的端口号
ssh -p <port> <username>@<IP address>

ssh -p 2222 user2@192.168.1.1
```


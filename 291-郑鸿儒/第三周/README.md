## python 格式
### 第一行#!/usr/bin/env python 称为shebang或者hashbang行（用于unix或类unix系统）
    1. 这个命令的意思是查找环境变量$path中列出的目录，并尝试找到并执行指定的命令
    2. 好处是可移植性好，可以查找并执行特定的解释器，而不需要指定解释器的绝对路径
    3. 本身不执行，只是用于搜索并执行其他命令
### shebang行下面或第一行 # encoding=编码格式 或 # _*_ coding: 编码格式 _*_
    1. 用于声明文档的编码格式，不声明有报错的可能，如 SyntaxError: Non-UTF-8 code starting with '\xd6' in file
        E:\AI\demo\histogram euqalization.py on line 17, but no encoding declared;
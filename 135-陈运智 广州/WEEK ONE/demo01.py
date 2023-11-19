# 输入两个数字相加
num1 = input("请输入数字1：")
num2 = input("请输入数字2：")
sum=int(num1)+int(num2)
print(sum)
# 输入两个数字相加（答案）
num1 = input("请输入数字1：")
num2 = input("请输入数字2：")
sum=int(num1)+int(num2)
print('数字{0}和{1}相加的结果：{2}'.format(num1,num2,sum))

# 生成随机数，这个是百度的
import random
random.randint(1,10)

# 九九乘法表格
for i in range(9):
    for j in range(9):
        i
        if i>=j:
            s=(i+1)*(j+1)
            print("%d"%(i+1),"*%d"%(j+1),"=%d"%s,end=" ")
        else:
            print()
# 九九乘法表格(答案）
for i in range(1,10):
    for j in range(1,i+1):
        print('{}x{}={}\t'.format(i,j,i*j),end='')
    print()


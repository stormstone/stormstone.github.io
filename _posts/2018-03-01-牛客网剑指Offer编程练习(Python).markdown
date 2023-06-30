---
layout:     post
title:      "牛客网剑指offer编程练习(Python)"
date:       2018-03-01 22:20:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - LeetCode
---

牛客网编程练习-剑指offer，python实现。


### 1.【链表】从尾到头打印链表
输入一个链表，从尾到头打印链表每个节点的值。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        r = []
        while listNode:
            r.insert(0,listNode.val)
            listNode = listNode.next
        return r
```

### 2.【二叉树】重建二叉树
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre) == 0:
            return None
        if len(pre) == 1:#不能用elif，进不来。。。
            return TreeNode(pre[0])
        else:
            flag = TreeNode(pre[0])
            # 迭代左右子树
            flag.left = self.reConstructBinaryTree(pre[1:tin.index(pre[0])+1],tin[:tin.index(pre[0])])
            flag.right = self.reConstructBinaryTree(pre[tin.index(pre[0])+1:],tin[tin.index(pre[0])+1:])
        return flag
```

### 3.【查找】旋转数组的最小数字
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

```python
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        length = len(rotateArray)
        if length == 0:
            return 0
        if length == 1:
            return rotateArray[0]
        left, right = 0, length - 1

        while left <= right:
            mid = (left + right) >> 1
            if rotateArray[mid] > rotateArray[right]:
                left = mid + 1
            elif rotateArray[mid] < rotateArray[right]:
                right = mid
            else:
                right -= 1
            if left >= right:
                break
        return rotateArray[left]

rotateArray = [3, 4, 5, 1, 2]
s = Solution()
print(s.minNumberInRotateArray(rotateArray))
```

### 4.【斐波那契】变态跳台阶
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
```python
class Solution:
def jumpFloorII(self, number):
    # write code here
    if number == 1:
        return 1
    else:
        return pow(2, number - 1)
```

### 5.【位运算】二进制中1的个数
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        return sum([(n>>i & 1) for i in range(0,32)])
```

### 6.【代码完整性】数值的整数次方 
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
```python
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        pow = 1.0000000
        if exponent < 0:
            for i in range(0-exponent):
                pow = pow * base
            return 1.0/pow
        else:
            for i in range(exponent):
                pow = pow * base
            return pow
```

### 7.【代码完整性】调整数组顺序使奇数位于偶数前面 
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        list1 = list()
        list2 = list()
        for i in array:
            if i % 2 == 1:
                list1.append(i)
            else:
                list2.append(i)
        return list1 + list2
```

### 8.【链表】链表中倒数第k个结点 
输入一个链表，输出该链表中倒数第k个结点。
```python
# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if head is None or k==0:
            return None
        p1 = head
        p2 = head
        for i in range(k-1):
            if p1.next:
                p1 = p1.next
            else:
                return None
        while p1.next:
            p1 = p1.next
            p2 = p2.next
        return p2
```

### 9.【链表】反转链表
输入一个链表，反转链表后，输出链表的所有元素。
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if not pHead or not pHead.next:
            return pHead
        l = None
        while pHead:
            tmp = pHead.next
            pHead.next = l
            l=pHead
            pHead = tmp
        return l
```

### 10.【链表】合并两个排序的链表 
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        mergeHead = ListNode(18)
        p = mergeHead
        while pHead1 and pHead2:
            if pHead1.val >= pHead2.val:
                mergeHead.next = pHead2
                pHead2 = pHead2.next
            else:
                mergeHead.next = pHead1
                pHead1 = pHead1.next
                  
            mergeHead = mergeHead.next
        if pHead1:
            mergeHead.next = pHead1
        elif pHead2:
            mergeHead.next = pHead2
        return p.next
```

